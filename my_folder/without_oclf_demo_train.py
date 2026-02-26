import os
import json
import math
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

from model import TextEncoder, SlotTextModel

import spacy
from tqdm import *
import random

from data import SlotCaptionDataset
            

@torch.no_grad()
def evaluate(model:nn.Module, loader:DataLoader, device: torch.device, args) -> Dict[str, float]:
    model.eval()
    total_loss, total_n = 0.0, 0
    total_attr_loss = 0.0
    total_scene_loss = 0.0

    correct, correct_attr, correct_scene = 0,0,0
    
    for batch in loader:
        slot_feats = batch["slot_feat"].to(device)  #1,M,D
        tok_ids = batch["caption_id"].to(device)
        tok_lens = batch["caption_len"].to(device)
        
        neg_tok_ids = [t.to(device) for t in batch["neg_tok_ids"]]
        neg_tok_lens = [t.to(device) for t in batch["neg_tok_lens"]]
        
        attribute_loss, scene_loss, loss = model.evaluate(
            slot_feats.repeat(4,1,1), 
            torch.stack([tok_ids] + neg_tok_ids).squeeze(1),
            torch.stack([tok_lens] + neg_tok_lens).squeeze(1)
        )
        
        total_loss += loss.mean().item()
        #total_attr_loss += attribute_loss.mean().item()
        #total_scene_loss += scene_loss.mean().item()
        
        pred = loss.argmin().item()
        #pred_attr = attribute_loss.argmin().item()
        #pred_scene = scene_loss.argmin().item()
        
        correct += int(pred == 0)
        #correct_attr += int(pred_attr == 0)
        #correct_scene += int(pred_scene == 0)
        
        total_n += 1
        
        if args.dev:
            break
    
    return {
        "loss": total_loss / max(total_n, 1),
        "acc_i2t_top1": correct / max(total_n, 1),
    }

def train(args):
    print(f"=============== Start {args.train_percent} percent train ===========")
    print("slot experiment")
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device: ", device)
    
    root_dir = args.data_root        

    with open(args.vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    train_set = SlotCaptionDataset(
        args,
        index_json=os.path.join(root_dir, "index_movi_a_train.json"),
        root_dir=root_dir, 
        split = "train", 
        train_percent = args.train_percent
    )
    val_set = SlotCaptionDataset(
        args,
        index_json=os.path.join(root_dir, "index_movi_a_train.json"),
        root_dir=root_dir, 
        split = "val"
    )

    # MODIFICATION
    print(f"len train set = {len(train_set)}")
    print(f"len val set = {len(val_set)}")
    
    train_set.preload_all_feats()
    val_set.preload_all_feats()
    
    d_slot = train_set[0]["slot_feat"].shape[-1]

    g = torch.Generator()
    g.manual_seed(42)
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle =True,
        #generator=g,
        num_workers = args.num_workers,
        pin_memory =True,
        # TODO: collate_fn = make_collate(tokenizer),
        drop_last =True,
        #collate_fn=collate_slot_caption,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle =False,
        num_workers = args.num_workers,
        pin_memory =True,
        drop_last =False,
        #collate_fn=collate_slot_caption,
        # persistent_workers=True, #
        # prefetch_factor=4,
    )

    textencoder =  TextEncoder()
    model = SlotTextModel(
        textencoder=textencoder,
        slot_dim=d_slot,
        embed_dim=256,
        tau=0.1,
    ).to(device)

    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    logdir = Path(args.logdir) / (time.strftime("%Y%m%d_%H%M%S") + "_" + str(args.train_percent))
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    global_step = 0
    best_val = math.inf
    
    
    if val_loader is not None:
        metrics = evaluate(model, val_loader, device, args)
        print(
            "eval with scratch model: "
            f"loss {metrics['loss']:.4f} "
            f"loss_attr {metrics['loss_attr']:.4f} "
            f"loss_scene {metrics['loss_scene']:.4f} "
            f"acc_i2t_top1 {metrics['acc_i2t_top1']:.4f} "
            f"acc_i2t_top1_attr {metrics['acc_i2t_top1_attr']:.4f} "
            f"acc_i2t_top1_scene {metrics['acc_i2t_top1_scene']:.4f}"
        )
    if args.eval:
        args.epochs = 0

    for epoch in range(args.epochs):
        if not args.eval:
            model.train()
            t0 = time.time()
            
            for it, batch in enumerate(train_loader):
                slot_feat = batch["slot_feat"].to(device)
                # captions = batch["captions"]
                # captions = [list(x) for x in zip(*captions)]
                #tok_ids   = batch["tok_ids"].to(device)        #B,10,25
                #tok_lens  = batch["tok_lens"].to(device)       #B,10
                tok_ids = batch["caption_id"].to(device)    #B,1,25
                tok_lens = batch["caption_len"].to(device)  #B,1
                
                
                attribute_loss, scene_loss, loss = model(slot_feat, tok_ids, tok_lens)
    
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
    
                if global_step % args.log_every == 0:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    print(f"epoch {epoch:02d} iter {it:05d} step {global_step:07d} loss {loss.item():.4f}")
                global_step += 1
                
                if args.dev:
                    break
            
            writer.add_scalar("train/epoch_time_sec", time.time() - t0, epoch)
            print("train/epoch_time_sec", time.time() - t0, epoch)
        
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, args)
            
            writer.add_scalar("val/loss", metrics["loss"], epoch)
            writer.add_scalar("val/acc_i2t_top1", metrics["acc_i2t_top1"], epoch)
            print(
                f"[val] epoch {epoch:02d} "
                f"loss {metrics['loss']:.4f} "
                f"acc_i2t_top1 {metrics['acc_i2t_top1']:.4f} "
            )


            if metrics["loss"] < best_val:
                best_val = metrics["loss"]
                save_path = logdir / "best_model.pth"
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": vars(args),
                    "d_slot": d_slot,
                }
                torch.save(ckpt, save_path)
                print(f"[ckpt] saved best.pt (val loss {best_val:.4f})")
        
        if (epoch + 1) % args.save_every == 0:
            save_path = logdir / f"model_epoch{epoch:02d}.pth"
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args),
                "d_slot": d_slot,
            }
            torch.save(ckpt, save_path)
            print(f"[ckpt] saved model_epoch{epoch:02d}.pth")
        
        if args.dev:
            break
    writer.close()
    print("Training completed.")



if __name__=="__main__":
    # dataset = SlotCaptionDataset(
    #     index_json="/home/wz3008/slot-attn/output/index_movi_a_train.json",
    #     root_dir="/home/wz3008/slot-attn/output"
    # )
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default ="/home/wz3008/slot-attn/output")
    p.add_argument("--vocab-path", type=str, default="/home/wz3008/slot-attn/vocab.json")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--logdir", type=str, default="./logs/")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--train_percent", type=float, default=0.9)
    
    p.add_argument("--dev", action="store_true")
    p.add_argument("--eval", action="store_true")
    

    args = p.parse_args()
    train(args)

    
