import re
import torch
from tqdm import *
import numpy as np

from pathlib import Path
import torchvision.utils as vutils
import torch.nn.functional as F

def basic_tokenize(text: str):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z\d]", text.lower())
    
    
def batch_to_cuda(batch: dict, device: str = "cuda"): 
    device = torch.device(device) 
    def move(x): 
        if torch.is_tensor(x): 
            return x.to(device, non_blocking=True) 
        if isinstance(x, dict): 
            return {k: move(v) for k, v in x.items()} 
        if isinstance(x, (list, tuple)): 
            y = [move(v) for v in x] 
            return type(x)(y) 
        return x 
    return move(batch)
    
"""
meme:
    username[int]
TypeError: 'type' object is not subscriptable
"""
def fixed_neg_indices(pos_idx, N):
    steps = (97, 193, 389)
    negs = []
    for s in steps:
        j = (pos_idx + s) % N
        if j == pos_idx or j in negs:
            j = (j + 1) % N
        if j == pos_idx or j in negs:
            j = (j + 2) % N
        negs.append(j)
    return negs
    
@torch.no_grad()
def build_token_pool_from_loader(loader):
    device = "cpu"
        #max_items = 20000
    tok_ids_list, tok_lns_list, keys = [], [], []
    #n = 0

    for i,batch in tqdm(enumerate(loader)):
        tok_ids = batch["instances"]["tok_ids"]  # [B,L]
        tok_lns = batch["instances"]["tok_lns"]  # [B]
        
        tok_ids = tok_ids.to(torch.long).to(device)
        tok_lns = tok_lns.to(torch.long).to(device)

            #B = tok_ids.shape[0]
            #for b in range(B):
            #    ids = tok_ids[b].detach().cpu()
            #    lns = tok_lns[b].detach().cpu()
            #    tok_ids_list.append(ids)
            #    tok_lns_list.append(lns)
            #    keys.append(tuple(ids.tolist()))
            #    n += 1
        tok_ids_list.append(tok_ids.to(torch.long).cpu())
        tok_lns_list.append(tok_lns.to(torch.long).cpu())

    pool_tok_ids = torch.stack(tok_ids_list, dim=0)  # [N,L]
    pool_tok_lns = torch.stack(tok_lns_list, dim=0)  # [N]
    return pool_tok_ids, pool_tok_lns
    

def mask_bool_video(x: torch.Tensor) -> torch.Tensor:
    B, T, K, C, H, W = x.shape
    assert C == 1

    flat = x.view(B, T, K, H * W)

    for i in range(B):
        for j in range(T):
            for n in range(H*W):
                max_idx = flat[i, j, :, n].argmax()
                flat[i, j, :, n] = 0
                flat[i, j, max_idx, n] = 1
    return flat.view(B, T, K, 1, H, W)

def mask_bool(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, K, H, W]
    return: [B, K, H, W] one-hot along K at each (B,H,W)
    """
    assert x.dim() == 4, f"expected [B,K,H,W], got {x.shape}"
    
    max_idx = x.argmax(dim=1, keepdim=True)              # [B, 1, H, W]
    
    out = torch.zeros_like(x)
    out.scatter_(dim=1, index=max_idx, value=1)
    return out

def is_background_batch(slot_masks: torch.Tensor,
                        area_thresh: float = 0.5,
                        edge_frac: float = 0.5) -> torch.Tensor:
    """
    slot_masks: [B, M, H, W]
    return: bg_mask [B, M] (bool)
    """
    B, M, H, W = slot_masks.shape
    
    slot_sum = slot_masks.sum(dim=(-1, -2))
    area_frac = slot_sum / float(H * W)
    
    top = slot_masks[:, :, 0, :].sum(dim=-1)             # [B, M]
    bottom = slot_masks[:, :, -1, :].sum(dim=-1)         # [B, M]
    left = slot_masks[:, :, :, 0].sum(dim=-1)            # [B, M]
    right = slot_masks[:, :, :, -1].sum(dim=-1)          # [B, M]
    edge_sum = top + bottom + left + right               # [B, M]
    
    edge_thresh = 2.0 * (H + W) * edge_frac
    bg = (area_frac > area_thresh) | (edge_sum > 2*(H+w)*0.5)
    return bg

def iou_bool(a:np.ndarray, b:np.ndarray, eps:float=1e-6)->float:
    intersection = np.logical_and(a,b).sum()
    union = np.logical_or(a,b).sum()
    return float(intersection) / float(union+eps)

def select_slot(slot_feats, slot_masks, gt_masks):
    # [8, 11, 256]
    # B, 11, 128, 128
    # B, 10, 1, 128, 128
    
    _, K, D = slot_feats.shape
    B = gt_masks.shape[0]
    H, W = slot_masks.shape[-2], slot_masks.shape[-1]
    
    slot_masks_oh = mask_bool(slot_masks)                       # [B,11,H,W]
    slot_masks_bool = slot_masks_oh.bool()
    
    sel_feats = torch.zeros_like(slot_feats)                    # [B,11,D]
    sel_masks = torch.zeros_like(slot_masks_oh)                 # [B,11,H,W]
    sel_idx   = torch.full((B, K), -1, device=slot_feats.device, dtype=torch.long)
    sel_M     = torch.zeros((B,), device=slot_feats.device, dtype=torch.long)
    
    # import pdb; pdb.set_trace()
    
    for b in range(B):
        gt_b = gt_masks[b, :, 0]                                # [10,H,W]
        valid = (gt_b.sum(dim=(-1, -2)) > 0)                    # [10] bool
        valid_ids = valid.nonzero(as_tuple=False).squeeze(1)    # [M]
        M = int(valid_ids.numel())
        sel_M[b] = M
        if M == 0:
            continue
            
        slots_np = slot_masks_bool[b].detach().cpu().numpy()    # [11,H,W] bool
        
        for t, gid in enumerate(valid_ids.tolist()):
            gt_np = gt_b[gid].detach().cpu().numpy().astype(bool)  # [H,W] bool
            
            ious = [iou_bool(gt_np, slots_np[k]) for k in range(K)]
            best_k = int(np.argmax(ious))
            
            sel_idx[b, t] = best_k
            sel_masks[b, t] = slot_masks_oh[b, best_k]          # [H,W]
            sel_feats[b, t] = slot_feats[b, best_k]             # [D]
            
    return sel_feats, sel_masks      #, sel_idx, sel_M

def steve_visualize(video, attns, num_slots, N=1):
    B, T, C, H, W = video.size()
    # B,T,K,C,H,W = attns.size()

    frames = []
    for t in range(T):
        video_t = video[:N, t, None, :, :, :] # N,1,C,H,W
        attns_t = attns[:N, t, :, :, :, :] #N,K,1,H,W
        
        scores_t = attns_t.squeeze(2)              # [N,K,H,W]
        idx = scores_t.argmax(dim=1)               # [N,H,W]
        hard = F.one_hot(idx, num_classes=num_slots).permute(0, 3, 1, 2).float()  # [N,K,H,W]
        hard = hard.unsqueeze(2)                   # [N,K,1,H,W]

        mask_video_t = video_t[:,0].unsqueeze(1) * hard   #attns_t.float()

        # tile
        tiles = torch.cat((video_t, mask_video_t), dim=1).flatten(end_dim=1)  # N*(K+1),C,H,W

        # grid
        frame = vutils.make_grid(tiles, nrow=(num_slots + 1), pad_value=0.8)
        frames += [frame]

    frames = torch.stack(frames, dim=0).unsqueeze(0)

    return frames

def visualize_mask_heat_map(input, out):
    video = input['image'].unsqueeze(1)    # [1, 1, 3, 224, 224]
    m = out['object_decoder'].masks # torch.Size([1, 11, 224, 224])
    m = m.unsqueeze(2).repeat(1,1,3,1,1)
    tiles = torch.cat([video[:,0].unsqueeze(1), m], dim=1).flatten(0,1)
    grid = vutils.make_grid(tiles, nrow=(m.shape[1]+1), pad_value=0.8)
    frame = vutils.make_grid(tiles, nrow=(11 + 1), pad_value=0.8)  # CxNxN
    vutils.save_image(frame, '/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/my_folder/visualize/mask_heat_map.jpg')
    
def visualize(args, model, val_loader):
    save_dir = Path('./visualize')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    for i, batch in enumerate(val_loader):
        if i % 1 == 0:
            if args.use_original_data:
                input = batch
            else:
                input = {
                    "batch_size": batch["image"].shape[0],
                    "image": batch["image"],   # [1, 3, 224, 224]
                    "instances": {
                        "tok_ids": batch["tok_ids"],
                        "tok_lns": batch["tok_lns"],
                    },
                    "gt_masks": batch.get("gt_masks", None),
                }
            input = batch_to_cuda(input)
            out = model.oclf_model(input)
            
            visualize_mask_heat_map(input, out)
            
            video = input['image'].unsqueeze(1)    # [1, 1, 3, 224, 224]
            slot_masks = out['object_decoder'].masks # torch.Size([1, 11, 224, 224])
            #slot_masks = slot_masks[:,1:9,:,:]
            attns = slot_masks.unsqueeze(1).unsqueeze(3)
            
            frame = steve_visualize(video, attns, 11)
            vutils.save_image(frame[0][0], f'/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/my_folder/visualize/test_{i}.jpg')
            import pdb; pdb.set_trace()
        
        
        
            
    
    
    
    
    
    