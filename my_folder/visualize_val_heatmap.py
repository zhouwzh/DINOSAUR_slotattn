import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import SlotTextModel
from textmodel import TextEncoder
from data import SlotCaptionDataset
from train import build_val_input
from get_oclf_model_demo import build_oclf_model_same_arch


def normalize_map(x: np.ndarray):
    x = x - x.min()
    if x.max() > 1e-8:
        x = x / x.max()
    return x


def denorm_img(x):
    x = x.detach().cpu().float()
    x = x.permute(1, 2, 0).numpy()
    x = np.clip(x, 0, 1)
    return x


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def make_top1_slot_binary_overlay(image_chw, slot_masks, slot_weights, alpha=0.5, threshold=0.5):
    """
    只可视化 attention 分数最高的那个 slot 对应的 mask

    Args:
        image_chw:   [3, H, W]
        slot_masks:  [K, 1, H, W] or [K, H, W]
        slot_weights:[K]
    Returns:
        img_np:      [H, W, 3]
        mask_np:     [H, W]          原始 top1 mask, 已归一化
        binary_np:   [H, W]          threshold 后的二值 mask
        overlay_np:  [H, W, 3]
        top_idx:     int
    """
    img_np = image_chw.detach().cpu().float().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    masks = slot_masks.detach().cpu().float()
    if masks.dim() == 4:
        masks = masks.squeeze(1)  # [K, H, W]
    masks = masks.numpy()

    weights = slot_weights.detach().cpu().float().numpy()
    top_idx = int(np.argmax(weights))

    mask_np = masks[top_idx]
    mask_np = normalize_map(mask_np)
    binary_np = (mask_np > threshold).astype(np.float32)

    overlay_np = img_np.copy()
    red = np.zeros_like(img_np)
    red[..., 0] = 1.0

    overlay_np = img_np * (1 - alpha * binary_np[..., None]) + red * (alpha * binary_np[..., None])
    overlay_np = np.clip(overlay_np, 0, 1)

    return img_np, mask_np, binary_np, overlay_np, top_idx


def load_model_and_ckpt(args, device):
    dm, lm = build_oclf_model_same_arch(
        args.train_config_path,
        args.checkpoint_path,
    )

    textencoder = TextEncoder()
    model = SlotTextModel(
        args=args,
        oclf_model=lm,
        textencoder=textencoder,
    ).to(device)

    ckpt = torch.load(args.vis_ckpt, map_location=device)
    state = ckpt["model"]

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                new_state[k[len("module."):]] = v
            else:
                new_state[k] = v
        model.load_state_dict(new_state, strict=True)

    model.eval()
    return model, dm


def build_val_loader(args, dm):
    if args.use_original_data:
        val_dataset = dm._create_webdataset(
            dm.val_shards,
            shuffle=False,
            n_datapoints=int(dm.val_size),
            keys_to_keep=("video", "instances", "segmentations"),
            transforms=None,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
    else:
        val_dataset = SlotCaptionDataset(
            args,
            split="val",
            index_json=os.path.join(args.data_root, "index_movi_a_train.json"),
            root_dir=args.data_root,
        )

        val_dataset.check_val_metadata()

        if args.pre_slot_feature:
            val_dataset.preload_all_feats()

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    return val_loader


def _get_target_caption(batch):
    if "target_caption" in batch:
        x = batch["target_caption"]
        return x[0] if isinstance(x, list) else x
    if "target_label" in batch:
        x = batch["target_label"]
        return x[0] if isinstance(x, list) else x
    return "N/A"


def visualize_one_batch(batch, model, args, save_path):
    with torch.no_grad():
        input_dict, target_index = build_val_input(batch, args)
        out = model(input_dict, mode="val_vis")

    scores = out["scores"].detach().cpu()          # [4]
    cls_attn = out["cls_attn"].detach().cpu()      # [4, L]
    pred_index = int(torch.argmax(scores).item())

    images = batch["images"].squeeze(0)            # [4, C, H, W]
    has_pre_slot_mask = ("pre_slot_mask" in batch) and (batch["pre_slot_mask"].numel() > 0)

    # token layout: [CLS] + 4 text + K slots
    L = cls_attn.shape[1]
    K = L - 1 - 4
    slot_attn = cls_attn[:, 5:5 + K]               # [4, K]

    fig = plt.figure(figsize=(18, 10))

    # ===== 第一行：4 张候选图 =====
    for i in range(4):
        ax = plt.subplot(3, 4, i + 1)
        ax.imshow(denorm_img(images[i]))

        title = f"img {i}\nscore={scores[i]:.3f}"
        if i == target_index:
            title += "\n[GT]"
        if i == pred_index:
            title += "\n[PRED]"
        ax.set_title(title)
        ax.axis("off")

    # ===== 第二行左：4-way score =====
    ax_bar = plt.subplot(3, 4, 5)
    ax_bar.bar(np.arange(4), scores.numpy())
    ax_bar.set_xticks(np.arange(4))
    ax_bar.set_xticklabels([f"img{i}" for i in range(4)])
    ax_bar.set_title("4-way scores")

    # ===== 第二行右：token-level heatmap =====
    ax_tok = plt.subplot(3, 4, (6, 8))
    token_labels = ["CLS", "txt1", "txt2", "txt3", "txt4"] + [f"slot{i}" for i in range(K)]
    im = ax_tok.imshow(cls_attn.numpy(), aspect="auto")
    ax_tok.set_yticks(np.arange(4))
    ax_tok.set_yticklabels([f"img{i}" for i in range(4)])
    ax_tok.set_xticks(np.arange(len(token_labels)))
    ax_tok.set_xticklabels(token_labels, rotation=45, ha="right")
    ax_tok.set_title("CLS attention to [CLS + text + slots]")
    plt.colorbar(im, ax=ax_tok, fraction=0.02, pad=0.02)

    # ===== 第三行：只画 GT 和 PRED 的 top-1 slot mask =====
    candidates_to_show = [target_index]
    if pred_index != target_index:
        candidates_to_show.append(pred_index)

    for j, img_idx in enumerate(candidates_to_show[:2]):
        if has_pre_slot_mask:
            # 预期 shape: [1, 4, K, 1, H, W] 或 [1, 4, K, H, W]
            pre_masks = batch["pre_slot_mask"].squeeze(0)[img_idx]

            # 只在 valid slots 里选 top1
            cur_weights = slot_attn[img_idx].clone()
            if "slot_mask" in batch and batch["slot_mask"].numel() > 0:
                valid_slot_mask = batch["slot_mask"].squeeze(0)[img_idx].bool()
                cur_weights[~valid_slot_mask[:K]] = -1e9

            img_np, mask_np, binary_np, overlay_np, top_idx = make_top1_slot_binary_overlay(
                images[img_idx],
                pre_masks[:K],
                cur_weights[:K],
                alpha=0.5,
                threshold=0.5,
            )

            tag = "GT" if img_idx == target_index else "PRED"

            ax1 = plt.subplot(3, 4, 9 + 2 * j)
            ax1.imshow(img_np)
            ax1.set_title(f"{tag} image {img_idx}")
            ax1.axis("off")

            ax2 = plt.subplot(3, 4, 10 + 2 * j)
            ax2.imshow(overlay_np)
            ax2.set_title(f"{tag} top-1 slot mask | slot={top_idx}")
            ax2.axis("off")
        else:
            ax = plt.subplot(3, 4, 9 + j)
            ax.text(
                0.5, 0.5,
                "No pre_slot_mask loaded.\nRun with --pre_slot_mask",
                ha="center", va="center", fontsize=12
            )
            ax.axis("off")

    target_caption = _get_target_caption(batch)
    plt.suptitle(
        f'target caption: "{target_caption}" | GT={target_index}, PRED={pred_index}',
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    model, dm = load_model_and_ckpt(args, device)
    val_loader = build_val_loader(args, dm)

    for i, batch in enumerate(val_loader):
        save_path = os.path.join(args.outdir, f"val_vis_{i:04d}.png")
        visualize_one_batch(batch, model, args, save_path)
        print(f"saved: {save_path}")

        if i + 1 >= args.max_samples:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis_ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./vis_out")
    parser.add_argument("--max_samples", type=int, default=10)

    parser.add_argument("--train_config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--use_original_data", action="store_true")

    parser.add_argument("--data-root", type=str, default="/output")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument(
        "--val_metadata_path",
        type=str,
        default="/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/my_folder/val_metadata.json",
    )
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--vit", action="store_true")
    parser.add_argument("--pre_slot_feature", action="store_true")
    parser.add_argument("--pre_slot_mask", action="store_true")
    parser.add_argument("--steve", action="store_true")
    parser.add_argument("--vit_patches", action="store_true")
    parser.add_argument("--num_slots", type=int, default=11)
    parser.add_argument("--criterion", type=str, default="")
    parser.add_argument("--tau", type=float, default=0.07)

    args = parser.parse_args()
    main(args)