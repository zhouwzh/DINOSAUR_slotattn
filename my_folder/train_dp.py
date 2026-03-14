import argparse
import torch
from pathlib import Path
import time
import os
import math
import torch.nn as nn
from torch.nn import DataParallel as DP
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import *

from model import SlotTextModel
from textmodel import TextEncoder
from data import SlotCaptionDataset
from my_utils import *
from get_oclf_model_demo import build_oclf_model_same_arch, MOVIA_LABEL_PREFIX
from ocl.datasets import _get_single_element_transforms
from ocl.datasets import _get_batch_transforms


class SimpleLogger:
    def __init__(self, mode: str, logdir: Path):
        self.mode = mode
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.tb_writer = None
        self.txt_file = None

        if self.mode == "tensorboard":
            self.tb_writer = SummaryWriter(self.logdir)
        elif self.mode == "txt":
            self.txt_file = self.logdir / "train_log.txt"
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


def unwrap_model(model):
    return model.module if isinstance(model, DP) else model


def build_run_name(args) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.dev:
        return (f"dev_{ts}")
    return (
        f"{ts}"
        f"_vit-{int(args.vit)}"
        f"_preslot-{int(args.pre_slot_feature)}"
        f"_tp-{args.train_percent}"
        f"_criterion-{args.criterion}"
    )


def using_dp(model):
    return isinstance(model, DP)

# =========================================================
# batch -> input
# =========================================================
def build_train_input(batch, args, move_to_cuda=True):
    """
    train batch keys:
    "video_idx", "frame_idx", "image", "gt_masks", "word_tokens", "slot_feat"
    """
    input_dict = {
        # "batch_size": batch["image"].shape[0],
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
    
    if move_to_cuda:
        input_dict = batch_to_cuda(input_dict)
    return batch_to_cuda(input_dict)


def build_val_input(batch, args, move_to_cuda=True):
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
    # images = batch["images"].squeeze(0)              # [4, C, H, W]
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

    if move_to_cuda:
        input_dict = batch_to_cuda(input_dict)
    return input_dict, target_index


# =========================================================
# evaluation
# =========================================================
def run_val_epoch(args, model, val_loader, logger=None, epoch=None, prefix="val", move_to_cuda=True):
    raw_model = unwrap_model(model)
    raw_model.eval()
    
    total_loss = 0.0
    total_n = 0
    correct = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_dict, target_index = build_val_input(batch, args, )
            
            scores = raw_model(input_dict, mode="val")
            #scores = model(input_dict, mode="val")
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
    raw_model = unwrap_model(model)
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model": raw_model.state_dict(),
        "opt": opt.state_dict(),
        "args": vars(args),
    }
    torch.save(ckpt, path)


# =========================================================
# train
# =========================================================
def train(args, model, train_loader, val_loader):
    run_name = build_run_name(args)
    logdir = Path(args.logdir) / run_name
    logdir.mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger(args.writer_type, logdir)

    global_step = 0
    best_val = math.inf

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.wd,
    )

    multi_gpu = isinstance(model, DP)

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
        move_to_cuda=multi_gpu,
    )

    # -----------------------------------------------------
    # main training loop
    # -----------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()

        for it, batch in enumerate(train_loader):
            input_dict = build_train_input(batch, args, move_to_cuda=multi_gpu)
            output = model(input_dict)

            if isinstance(output, tuple):
                loss, loss_b = output
            else:
                loss = output
                loss_b = None

            opt.zero_grad()
            if loss.dim() > 0:
                loss = loss.mean()
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
                move_to_cuda=multi_gpu,
            )
        
        if torch.cuda.is_available() and global_step % args.log_every == 0:
            for d in range(torch.cuda.device_count()):
                print(
                    f"[MEM] gpu{d} alloc={torch.cuda.memory_allocated(d)/1024**3:.2f}GB "
                    f"reserved={torch.cuda.memory_reserved(d)/1024**3:.2f}GB"
                )

        #     if metrics["loss"] < best_val:
        #         best_val = metrics["loss"]
        #         save_path = logdir / "best_model.pth"
        #         save_checkpoint(save_path, epoch, global_step, model, opt, args)
        #         logger.log_text(f"[ckpt] saved best_model.pth (val loss {best_val:.4f})")

        # if (epoch + 1) % args.save_every == 0:
        #     save_path = logdir / f"model_epoch{epoch:02d}.pth"
        #     save_checkpoint(save_path, epoch, global_step, model, opt, args)
        #     logger.log_text(f"[ckpt] saved model_epoch{epoch:02d}.pth")

        if args.dev:
            break

    logger.close()
    print("Training completed.")


def main(args):
    dm, lm = build_oclf_model_same_arch(
        args.train_config_path,
        args.checkpoint_path,
    )

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

        if args.pre_slot_feature:
            train_dataset.preload_all_feats()
            val_dataset.preload_all_feats()

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
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

    if device.type == "cuda":
        n_gpu = torch.cuda.device_count()
        print(f"Detected {n_gpu} GPU(s).")
        #print("Do not use DataParallel because oclf_model uses feature hooks.")
        if n_gpu > 1:
            print(f"Using DataParallel on {n_gpu} GPUs")
            model = DP(model)

    print(f"=============== Start {args.train_percent} percent train ===============")
    print(f"train data num: {len(train_dataset)}, val data num: {len(val_dataset)}")

    if args.visualize:
        visualize(args, unwrap_model(model), val_loader)

    train(args, model, train_loader, val_loader)


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--data-root", type=str, default="/output")
    p.add_argument("--vocab-path", type=str, default="/scratch/wz3008/new_SlotAttn/slot_attn_new/vocab.json")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")

    p.add_argument(
        "--logdir",
        type=str,
        default="/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/my_folder/logs/",
    )
    p.add_argument("--writer_type", type=str, default="tensorboard", choices=["tensorboard", "txt"])

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
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
    
    p.add_argument("--val_metadata_path", type=str, default="/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/my_folder/val_metadata.json")

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