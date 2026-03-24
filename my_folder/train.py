import argparse
import torch
from pathlib import Path
import time
import os
import math
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import *
import random
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from model import SlotTextModel
from textmodel import TextEncoder
from data import SlotCaptionDataset
from my_utils import *
from get_oclf_model_demo import build_oclf_model_same_arch, MOVIA_LABEL_PREFIX
from ocl.datasets import _get_single_element_transforms
from ocl.datasets import _get_batch_transforms

def _to_img_np(x):
    # x: [C,H,W]
    x = x.detach().cpu().clamp(0, 1)
    return x.permute(1, 2, 0).numpy()

def _to_mask_np(x):
    # x: [1,H,W] or [H,W]
    x = x.detach().cpu()
    if x.dim() == 3:
        x = x.squeeze(0)
    x = x.float()
    if x.max() > 0:
        x = x / x.max()
    return x.numpy()

def visualize_steve(args, model, train_loader):
    if not args.steve:
        raise RuntimeError("visualize_steve() currently only supports --steve")
    model.eval()
    save_root = Path("/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/my_folder/visualization")
    save_root.mkdir(parents=True, exist_ok=True)
    max_batches = getattr(args, "visualize_batches", 10)
    n_show = getattr(args, "visualize_n", 8)
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            input_dict = build_train_input(batch, args)
            images = input_dict["image"]   # [B, C, H, W]
            video, recon_dvae, recon_tf, attns_vis = model.visualize_steve(images,tau=0.1,hard=True,)
            # video:      [B, 1, C, H, W]
            # recon_dvae: [B, 1, C, H, W]
            # recon_tf:   [B, 1, C, H, W]
            # attns_vis:  [B, 1, K, C, H, W]
            B, T, C, H, W = video.shape
            N = min(n_show, B)    
            for t in range(T):
                video_t = video[:N, t, None, :, :, :]
                recon_dvae_t = recon_dvae[:N, t, None, :, :, :]
                recon_tf_t = recon_tf[:N, t, None, :, :, :]
                attns_t = attns_vis[:N, t, :, :, :, :]

                tiles = torch.cat((video_t, recon_dvae_t, recon_tf_t, attns_t),dim=1).flatten(end_dim=1)
                frame = vutils.make_grid(tiles,nrow=(model.steve_model.num_slots + 3),pad_value=0.8,)

                out_path = save_root / f"train_batch{batch_idx:04d}_frame{t:02d}.png"
                vutils.save_image(frame, out_path)

            print(f"[visualize] saved batch {batch_idx} to {save_root}")

            if batch_idx + 1 >= max_batches:
                break

def visualize_eval(args, model, val_loader):
    model.eval()

    save_root = Path("/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/my_folder/visualization/val_vis")
    save_root.mkdir(parents=True, exist_ok=True)

    max_samples = 50
    text_len = 4
    cls_offset = 1
    slot_start = cls_offset + text_len   # CLS + 4 text tokens

    with torch.no_grad():
        for sample_idx, batch in enumerate(val_loader):
            if sample_idx >= max_samples:
                break

            input_dict, target_index = build_val_input(batch, args)
            out = model(input_dict, mode="val_vis")

            scores = out["scores"].detach().cpu()           # [4]
            cls_attn = out["cls_attn"].detach().cpu()       # [4, 1+4+K]
            slot_mask = out["slot_mask"]
            vis_masks = out["vis_masks"]                    # [4,K,1,H,W] or None

            if slot_mask is not None:
                slot_mask = slot_mask.detach().cpu()
            if vis_masks is not None:
                vis_masks = vis_masks.detach().cpu()

            images = batch["images"].squeeze(0).detach().cpu()           # [4,C,H,W]
            contains_target = batch["contains_target"].squeeze(0).cpu()  # [4]
            pred_index = int(scores.argmax().item())

            if pred_index != target_index:
                continue

            target_caption = batch["target_label"][0]
            # batch_size=1 时 target_label 会被 DataLoader 包成长度1的 list

            # fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            # axes = axes.flatten()
            fig, axes = plt.subplots(4, 2, figsize=(10, 18))

            for i in range(4):
                ax_img = axes[i, 0]
                ax_mask = axes[i, 1]

                img_np = _to_img_np(images[i])

                cur_score = float(scores[i].item())
                is_pos = int(contains_target[i].item())

                cur_attn = cls_attn[i]              # [1+4+K]
                slot_attn = cur_attn[slot_start:]   # [K]

                if slot_mask is not None:
                    valid_mask = slot_mask[i].bool()
                    masked_slot_attn = slot_attn.clone()
                    masked_slot_attn[~valid_mask] = -1e9
                    best_slot_idx = int(masked_slot_attn.argmax().item())
                    best_slot_attn = float(masked_slot_attn[best_slot_idx].item())
                else:
                    best_slot_idx = int(slot_attn.argmax().item())
                    best_slot_attn = float(slot_attn[best_slot_idx].item())

                tags = []
                if i == target_index:
                    tags.append("GT")
                if i == pred_index:
                    tags.append("PRED")
                if is_pos == 1:
                    tags.append("POS")
                else:
                    tags.append("NEG")

                title = (
                    f"img {i} | match_score={cur_score:.4f}\n"
                    f"top_slot={best_slot_idx} | cls->slot_attn={best_slot_attn:.4f}\n"
                    f"{' | '.join(tags)}"
                )

                # 左边：原图
                ax_img.imshow(img_np)
                ax_img.axis("off")
                ax_img.set_title(title, fontsize=10)

                # 右边：mask
                if vis_masks is not None:
                    best_mask = vis_masks[i, best_slot_idx]   # [1,H,W]
                    mask_np = _to_mask_np(best_mask)
                    ax_mask.imshow(mask_np, cmap="jet", alpha=0.35 * (mask_np > 0.1))
                else:
                    ax_mask.text(0.5, 0.5, "No mask", ha="center", va="center", fontsize=14)

                ax_mask.axis("off")
                ax_mask.set_title(f"img {i} top-slot mask", fontsize=10)

                if i == pred_index:
                    for spine in ax_img.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor("red")
                        spine.set_linewidth(4)

                elif i == target_index:
                    for spine in ax_img.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor("lime")
                        spine.set_linewidth(4)

            fig.suptitle(
                f"val sample {sample_idx:04d}\n"
                f"target_caption: {target_caption}",
                fontsize=14
            )
            fig.tight_layout(rect=[0, 0, 1, 0.94])

            out_path = save_root / f"val_sample_{sample_idx:04d}.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            print(f"[visualize_eval] saved {out_path}")

class SimpleLogger:
    def __init__(self, mode: str, logdir: Path, run_name: str):
        self.mode = mode
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name


        self.tb_writer = None
        self.txt_file = None

        if self.mode == "tensorboard":
            self.tb_writer = SummaryWriter(self.logdir)
        elif self.mode == "txt":
            self.txt_file = self.logdir / f"{self.run_name}_train_log.txt"
            with open(self.txt_file, "a", encoding="utf-8") as f:
                f.write("===== training log start =====\n")
        else:
            raise ValueError(f"Unsupported writer mode: {self.mode}")
        
    def log_scalar(self, tag: str, value: float, step: int):
        if self.mode == "tensorboard":
            self.tb_writer.add_scalar(tag, value, step)
        elif self.mode == "txt":
            with open(self.txt_file, "a", encoding="utf-8") as f:
                f.write(f"[scalar] step={step} {tag}={value}\n")
    
    def log_text(self, message: str):
        print(message)
        if self.mode == "txt":
            with open(self.txt_file, "a", encoding="utf-8") as f:
                f.write(message + "\n")
    
    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.close()

def build_run_name(args) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.dev:
        return (f"dev_{ts}")
    return (
        f"{ts}"
        f"_steve-{int(args.steve)}"
        f"_vit-{int(args.vit)}"
        f"_preslot-{int(args.pre_slot_feature)}"
        f"_tp-{args.train_percent}"
        f"_criterion-{args.criterion}"
        f"_seed-{args.seed}"
    )

# =========================================================
# batch -> input
# =========================================================
def build_train_input(batch, args):
    """
    train batch keys:
    "video_idx", "frame_idx", "image", "gt_masks", "word_tokens", "slot_feat"
    """
    input_dict = {
        "batch_size": batch["image"].shape[0],
        "image": batch["image"],
        "instances": None,
        "gt_masks": batch["gt_masks"],
        "word_tokens": batch["word_tokens"],
        "caption_mask": batch["caption_mask"],
    }
    if args.pre_slot_feature:
        input_dict["slot_feat"] = batch["slot_feat"]
        input_dict["slot_mask"] = batch["slot_mask"]
    if args.pre_slot_mask:
        input_dict["pre_slot_mask"] = batch["pre_slot_mask"]
    return batch_to_cuda(input_dict)

def build_val_input(batch, args):
    """
    val batch keys:
    "word_tokens", "target_label", "target_index", "images", "contains_target", "slot_feats"

    DataLoader(batch_size=1):
    - images:        [1, 4, C, H, W]
    - word_tokens:   [1, 4]
    - target_index:  [1]
    - contains_target:[1, 4]
    - slot_feats:    [1, 4, M, D]  or [1, 4, 0]
    """
    images = batch["images"].squeeze(0)              # [4, C, H, W]
    word_tokens = batch["word_tokens"].squeeze(0)    # [4]
    target_index = batch["target_index"].item()

    word_tokens = word_tokens.unsqueeze(0).repeat(4, 1)  # [4, 4]

    input_dict = {
        "batch_size": 4,
        "image": images,
        "instances": None,
        "word_tokens": word_tokens,
    }
    if args.pre_slot_feature:
        input_dict["slot_feat"] = batch["slot_feats"].squeeze(0)
        input_dict["slot_mask"] = batch["slot_masks"].squeeze(0)
    if args.pre_slot_mask:
        input_dict["pre_slot_mask"] = batch["pre_slot_mask"].squeeze(0)
    input_dict = batch_to_cuda(input_dict)
    return input_dict, target_index

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================================================
# evaluation
# =========================================================
def run_val_epoch(args, model, val_loader, logger=None, epoch=None, prefix="val"):
    model.eval()
    total_loss = 0.0
    total_n = 0
    correct = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_dict, target_index = build_val_input(batch, args)
            scores = model(input_dict, mode="val")
            pred = scores.argmax().item()
            correct += int(pred == target_index)
            total_n += 1

            if args.dev:
                break

    metrics = {
        "acc_i2t_top1": correct / max(total_n, 1),
    }      
    if logger is not None and epoch is not None:
        logger.log_scalar(f"{prefix}/acc_i2t_top1", metrics["acc_i2t_top1"], epoch)
    msg = (
        f"[{prefix}] epoch {epoch if epoch is not None else -1:02d} "
        f"acc_i2t_top1 {metrics['acc_i2t_top1']:.4f}"
    )
    if logger is not None:
        logger.log_text(msg)
    else:
        print(msg)

    return metrics

# =========================================================
# checkpoint
# =========================================================
def save_checkpoint(path: Path, epoch: int, global_step: int, model, opt, args):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "args": vars(args),
    }
    torch.save(ckpt, path)

def load_resume_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"[resume] checkpoint loaded from {ckpt_path}")
    print(f"[resume] keys: {list(ckpt.keys())}")
    return ckpt

def resume_training(model, opt, ckpt_path: str, strict: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    msg = model.load_state_dict(ckpt["model"], strict=strict)
    opt.load_state_dict(ckpt["opt"])

    start_epoch = ckpt.get("epoch", -1) + 1
    global_step = ckpt.get("global_step", 0)

    print(f"[resume] loaded from {ckpt_path}")
    print(f"[resume] strict={strict}")
    print(f"[resume] start_epoch={start_epoch}")
    print(f"[resume] global_step={global_step}")
    print(f"[resume] missing_keys={msg.missing_keys}")
    print(f"[resume] unexpected_keys={msg.unexpected_keys}")

    return start_epoch, global_step

# =========================================================
# train
# =========================================================
def train(args, model, train_loader, val_loader, resume_ckpt=None):

    run_name = build_run_name(args)
    print(f"Run name: {run_name}")
    logdir = Path(args.logdir) / run_name
    logdir.mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger(args.writer_type, logdir, run_name)

    # global_step = 0
    # best_val = math.inf
    # best_acc = 0.0

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.wd,
    )
    
    start_epoch = 0
    global_step = 0
    best_acc = 0.0
    if resume_ckpt is not None:
        opt.load_state_dict(resume_ckpt["opt"])
        start_epoch = resume_ckpt.get("epoch", -1) + 1
        global_step = resume_ckpt.get("global_step", 0)

        print(f"[resume] start_epoch={start_epoch}")
        print(f"[resume] global_step={global_step}")
    # -----------------------------------------------------
    # evaluate with scratch model
    # -----------------------------------------------------
    logger.log_text("evaluate with scratch model")
    scratch_metrics = run_val_epoch(
        args=args,
        model=model,
        val_loader=val_loader,
        logger=logger,
        epoch=0,
        prefix="val_scratch",
    )
    best_acc = scratch_metrics["acc_i2t_top1"]

    # -----------------------------------------------------
    # main training loop
    # -----------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()

        for it, batch in enumerate(train_loader):
            input_dict = build_train_input(batch, args)
            output = model(input_dict)
            if isinstance(output, tuple):
                loss, loss_b = output
            else:
                loss = output
                loss_b = None

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            if global_step % args.log_every == 0:
                logger.log_scalar("train/loss", loss.item(), global_step)
                logger.log_text(
                    f"epoch {epoch:02d} iter {it:05d} step {global_step:07d} loss {loss.item():.4f}"
                )
            global_step += 1

            if args.dev:
                break

        epoch_time = time.time() - t0
        logger.log_scalar("train/epoch_time_sec", epoch_time, epoch)
        logger.log_text(f"train/epoch_time_sec epoch={epoch:02d} time={epoch_time:.4f}")

        if (epoch + 1) % args.val_every == 0:
            metrics = run_val_epoch(
                args=args,
                model=model,
                val_loader=val_loader,
                logger=logger,
                epoch=epoch,
                prefix="val",
            )

            if metrics["acc_i2t_top1"] > best_acc:
                best_acc = metrics["acc_i2t_top1"]
                save_path = logdir / "best_model.pth"
                save_checkpoint(save_path, epoch, global_step, model, opt, args)
                logger.log_text(f"[ckpt] saved best_model.pth (val acc {best_acc:.4f}, epoch {epoch if epoch is not None else -1:02d})")

        # if (epoch + 1) % args.save_every == 0:
        #     save_path = logdir / f"model_epoch{epoch:02d}.pth"
        #     save_checkpoint(save_path, epoch, global_step, model, opt, args)
        #     logger.log_text(f"[ckpt] saved model_epoch{epoch:02d}.pth")

        if args.dev:
            break

    logger.close()
    print("Training completed.")

def main(args):
    set_seed(args.seed)
    g  = torch.Generator()
    g.manual_seed(args.seed)
    
    dm, lm = build_oclf_model_same_arch(
        args.train_config_path,
        args.checkpoint_path,
    )
    # dm, lm = None, None

    if args.use_original_data:
        train_dataset = dm._create_webdataset(
            dm.train_shards,
            shuffle=dm.shuffle_train,
            n_datapoints=int(dm.train_size * args.train_percent),
            keys_to_keep=("video", "instances", "segmentations"),
            transforms=_get_single_element_transforms(dm.train_transforms),
        )

        val_dataset = dm._create_webdataset(
            dm.val_shards,
            shuffle=False,
            n_datapoints=int(dm.val_size),
            keys_to_keep=("video", "instances", "segmentations"),
            transforms=_get_single_element_transforms(dm.train_transforms),
        )

        train_loader = dm._create_dataloader(
            dataset=train_dataset,
            batch_transforms=_get_batch_transforms(dm.train_transforms),
            size=int(dm.train_size * args.train_percent),
            batch_size=dm.batch_size,
            partial_batches=False,
        )

        val_loader = dm._create_dataloader(
            dataset=val_dataset,
            batch_transforms=_get_batch_transforms(dm.train_transforms),
            size=int(dm.val_size),
            batch_size=1,
            partial_batches=False,
            num_workers=0,
        )
    else:
        train_dataset = SlotCaptionDataset(
            args,
            index_json=os.path.join(args.data_root, "index_movi_a_train.json"),
            root_dir=args.data_root,
            split="train",
            train_percent=args.train_percent,
        )

        val_dataset = SlotCaptionDataset(
            args,
            index_json=os.path.join(args.data_root, "index_movi_a_train.json"),
            root_dir=args.data_root,
            split="val",
        )

        val_dataset.check_val_metadata()

        # if args.pre_slot_feature:
        #     train_dataset.preload_all_feats()
        #     val_dataset.preload_all_feats()

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            generator=g,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    textencoder = TextEncoder()
    model = SlotTextModel(
        args,
        oclf_model=lm,
        textencoder=textencoder,
    ).to(device)

    resume_ckpt = None
    if args.resume_path:
        resume_ckpt = load_resume_ckpt(args.resume_path)
        msg = model.load_state_dict(resume_ckpt["model"], strict=True)
        print(f"[resume] model weights loaded in main()")
        print(f"[resume] missing_keys={msg.missing_keys}")
        print(f"[resume] unexpected_keys={msg.unexpected_keys}")
        print(f"=============== Start {args.train_percent} percent train ===============")
        print(f"train data num: {len(train_dataset)}, val data num: {len(val_dataset)}")

    if args.visualize:
        # if args.steve:
        #     visualize_steve(args, model, train_loader)
        visualize_eval(args, model, val_loader)
        return       

    train(args, model, train_loader, val_loader,resume_ckpt=resume_ckpt)

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--data_root", type=str, default="/output")
    p.add_argument("--vocab_path", type=str, default="/scratch/wz3008/new_SlotAttn/slot_attn_new/vocab.json")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")

    p.add_argument(
        "--logdir",
        type=str,
        default="/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/my_folder/logs/",
    )
    p.add_argument("--writer_type", type=str, default="tensorboard", choices=["tensorboard", "txt"])

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5)

    p.add_argument("--train_percent", type=float, default=0.9)
    p.add_argument("--image_size", type=int, default=128)

    p.add_argument("--dev", action="store_true")

    p.add_argument("--vit", action="store_true")
    p.add_argument("--use_slot_select", action="store_true")

    p.add_argument("--train_config_path", type=str, default=None)
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default="/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/outputs/slot_attention/movi_a/2026-02-02_05-55-06/",
    )

    p.add_argument("--visualize", action="store_true")
    p.add_argument("--use_original_data", action="store_true")
    p.add_argument("--pre_slot_feature", action="store_true")
    p.add_argument("--pre_slot_mask", action="store_true")
    p.add_argument("--vit_patches", action="store_true")
    p.add_argument("--steve", action="store_true")
    p.add_argument("--criterion", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    
    p.add_argument("--val_metadata_path", type=str, default="/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/my_folder/val_metadata.json")

    p.add_argument("--resume_path", type=str, default="")   #xxx.pth
    # p.add_argument("--load_model_path", type=str, default="")
    # p.add_argument("--strict_load", action="store_true")

    args = p.parse_args()

    if args.steve:
        args.image_size = 128

    if not args.train_config_path:
        ckpt_root = Path(args.checkpoint_path)
        args.train_config_path = ckpt_root / "config" / "config.yaml"
        if not args.train_config_path.is_file():
            raise FileNotFoundError(f"config not found: {args.train_config_path}")

        ckpt_dir = ckpt_root / "lightning_logs" / "version_0" / "checkpoints"
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")

        ckpt_files = sorted([p for p in ckpt_dir.iterdir() if p.is_file()])
        if len(ckpt_files) == 0:
            raise FileNotFoundError(f"no checkpoint files found in: {ckpt_dir}")

        args.checkpoint_path = ckpt_files[-1]

    print(f"Using checkpoint {args.checkpoint_path}")
    main(args)  
