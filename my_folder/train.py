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

from model import SlotTextModel, TextEncoder
from data import SlotCaptionDataset
from my_utils import *
from get_oclf_model_demo import build_oclf_model_same_arch, MOVIA_LABEL_PREFIX
from ocl.datasets import _get_single_element_transforms
from ocl.datasets import _get_batch_transforms


def train(args, model, train_loader, val_loader):
    
    logdir = Path(args.logdir) / (time.strftime("%Y%m%d_%H%M%S") + "_" + str(args.train_percent))
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    
    global_step = 0
    best_val = math.inf
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, 
        weight_decay=args.wd
    )
    
    # pool_tok_ids, pool_tok_lns = build_token_pool_from_loader(val_loader)
    # print("FULL val token pool size:", pool_tok_ids.shape)
    
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        
        for it, batch in enumerate(train_loader):
            input = {
                "batch_size": args.batch_size,
                "image": batch['image'],
                "instances":{
                    "tok_ids": batch['tok_ids'],
                    "tok_lns": batch['tok_lns'],
                },
                "gt_masks": batch["gt_masks"],
            }
            input = batch_to_cuda(input)
            
            loss, _ = model(input)
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            
            if global_step % args.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                print(f"epoch {epoch:02d} iter {it:05d} step {global_step:07d} loss {loss.item():.4f}")
            global_step += 1
            
            if args.dev: break
        
        writer.add_scalar("train/epoch_time_sec", time.time() - t0, epoch)
        print("train/epoch_time_sec", time.time() - t0, epoch)
        
        
        if (epoch+1) % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                # N = pool_tok_ids.shape[0]
                total_loss, total_n, correct = 0.0, 0, 0
                for pos_idx, batch in enumerate(val_loader):
                    
                    #pos_ids = batch["instances"]["tok_ids"]    # 1, 75
                    #pos_lns = batch["instances"]["tok_lns"]    # 1
                    #neg_idxs = fixed_neg_indices(pos_idx, N)
                    #neg_ids = pool_tok_ids[neg_idxs].to(pos_ids.device, non_blocking=True)   # 3, 1, 75
                    #neg_lns = pool_tok_lns[neg_idxs].to(pos_lns.device, non_blocking=True)   # 3, 1
                    #cand_ids = torch.cat([pos_ids, neg_ids.squeeze(1)], 0)  # [4,L]
                    #cand_lns = torch.cat([pos_lns, neg_lns.squeeze(1)], 0)  # [4]
                    
                    input = {
                        "instances":{
                            "tok_ids": torch.cat([batch['tok_ids'], batch['neg_tok_ids'].squeeze(0)], dim=0),
                            "tok_lns": torch.cat([batch['tok_lns'], batch['neg_tok_lns'].squeeze(0)], dim=0),
                        },
                        "image": batch['image'].repeat(4,1,1,1),
                        "batch_size": 4,
                        "gt_masks": batch["gt_masks"],
                    }
                    #batch["instances"]["tok_ids"] = cand_ids
                    #batch["instances"]["tok_lns"] = cand_lns
                    #batch['image'] = batch['image'].repeat(4,1,1,1)
                    #batch['batch_size'] = 4
                    
                    input = batch_to_cuda(input)
                    loss, loss_b = model(input)
                    
                    total_loss += loss.item()
                    pred = loss_b.argmin().item()
                    correct += int(pred == 0)
                    total_n += 1
                    
                    if args.dev: break
                
                metrics = {
                    "loss": total_loss / max(total_n, 1),
                    "acc_i2t_top1": correct / max(total_n, 1),
                }
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
            }
            torch.save(ckpt, save_path)
            print(f"[ckpt] saved model_epoch{epoch:02d}.pth")
        
        if args.dev:
            break    
            
    writer.close()
    print("Training completed.")
                
        
                        
        

def main(args):
    dm, lm = build_oclf_model_same_arch(
        args.train_config_path,
        args.checkpoint_path,
    )
    
    if args.use_original_data:
        # dm.train_size = 87327:  sample num after
        # SampleSlices(n_slices_per_input=9, fields=[image]), 
        # 10914 * 8 + 15(drop_last) = 87327
        train_dataset = dm._create_webdataset(
            dm.train_shards,
            shuffle=dm.shuffle_train,
            n_datapoints=int(dm.train_size * args.train_percent),
            keys_to_keep=("video", "instances", "segmentations"),
            transforms=_get_single_element_transforms(dm.train_transforms),
        )
        
        # dm.val_size = 6000
        # 6000 * 3 / 8 = 2250
        val_dataset = dm._create_webdataset(
            dm.val_shards,
            shuffle=False, #dm.shuffle_train,
            n_datapoints=int(dm.val_size),
            keys_to_keep=("video", "instances", "segmentations"),
            transforms=_get_single_element_transforms(dm.train_transforms),
        )
        #len(val_dataset) = 6000
        
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
            batch_size= 1 ,#dm.batch_size,      
            partial_batches=False,
            num_workers=0
        )
    
    # n = 0
    # for batch in enumerate(train_loader):  n += 1
    # print(f"train: {n}")
    # 10914
    # n = 0
    # for batch in enumerate(val_loader):  n += 1
    # print(f"val: {n}")
    # 2250
    
    # len = 185200
    else:
        train_dataset = SlotCaptionDataset(
            args,
            index_json = os.path.join(args.data_root , "index_movi_a_train.json"),
            root_dir = args.data_root,
            split = "train",
            train_percent = args.train_percent
        )
        
        val_dataset = SlotCaptionDataset(
            args,
            index_json=os.path.join(args.data_root, "index_movi_a_train.json"),
            root_dir=args.data_root, 
            split = "val"
        )
    
        # train_dataset.preload_all_feats()
        # val_dataset.preload_all_feats()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle =True,
            num_workers = args.num_workers,
            pin_memory =True,
            drop_last =True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle =False,
            num_workers = args.num_workers,
            pin_memory =True,
            drop_last =False,
        )
    
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    textencoder =  TextEncoder()
    model = SlotTextModel(
        args,
        oclf_model=lm,
        textencoder=textencoder,
    ).to(device)
    
    print(f"=============== Start {args.train_percent} percent train:  ===========")
    print(f"train data num: {len(train_dataset)}, val data num: {len(val_dataset)}")
    
    if args.visualize:
        visualize(args, model, train_loader)
        import sys; sys.exit(0)
    
    train(args, model, train_loader, val_loader)
    


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default ='/output')
    p.add_argument("--vocab-path", type=str, default='/scratch/wz3008/new_SlotAttn/slot_attn_new/vocab.json')
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--logdir", type=str, default="/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/my_folder/logs/")
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
    # p.add_argument("--checkpoint_path", type=str, default="/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/outputs/slot_attention/movi_a/2026-02-02_05-55-06/lightning_logs/version_0/checkpoints/epoch=34-step=381993.ckpt")
    p.add_argument("--checkpoint_path", type=str, default="/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/outputs/slot_attention/movi_a/2026-02-02_05-55-06/")
    
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--use_original_data", action="store_true")

    args = p.parse_args()
    if not args.train_config_path:
        dir = Path(args.checkpoint_path)
        args.train_config_path = dir / "config" / "config.yaml"
        if not args.train_config_path.is_file():
            raise FileNotFoundError(f"config not found: {train_config_path}")
            
        ckpt_dir = dir  / "lightning_logs" / "version_0" / "checkpoints"
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")
        
        args.checkpoint_path = sorted([p for p in ckpt_dir.iterdir() if p.is_file()])[-1]
    
    print(f"Using checkpoint {args.checkpoint_path}")
    main(args)

    
