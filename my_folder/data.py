import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import numpy as np

from my_utils import basic_tokenize


class SlotCaptionDataset(Dataset):
    def __init__(self, args, index_json: str, root_dir: str, split: str, train_percent: float = 0.9):
        self.root_dir = Path(root_dir)
        self.index_path = Path(index_json)
        self.split = split
        self.train_percent = train_percent
        groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
            # groups = {
            #     (0, 0): [rec1, rec2, rec3, ...],
            #     (0, 1): [rec9, rec10, ...],
            #     (0, 2): [rec17, ...],
            # }
        
        with open("/scratch/wz3008/new_SlotAttn/slot_attn_new/vocab.json") as f:
            self.vocab = json.load(f)

        self.sos = self.vocab["<sos>"]
        self.eos = self.vocab["<eos>"]
        self.pad = self.vocab["<pad>"]
        self.unk = self.vocab["<unk>"]
        self.max_seq_len = 75
        
        # small cache to avoid repeated spaCy on same captions (very helpful)
        self._tok_cache: Dict[str, Tuple[List[int], int]] = {}
        
        # {"video_idx": 0, "frame_idx": 0, "instance_idx": 1, "image_path": "00000000/00000000_image.png", "movi_mask_path": "00000000/00000000_mask_01>
        self.image_size = args.image_size
        self.img_tf = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(self.image_size),
            T.ToTensor(),  # float32 in [0, 1], shape (3,H,W)
        ])
        
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (int(rec["video_idx"]), int(rec["frame_idx"]))   # each key to multiple records which means multiple object in the frame
                groups.setdefault(key,[]).append(rec)
                
        self.frame_keys = sorted(groups.keys())

        n_total = len(self.frame_keys)
        n_train = int(n_total * self.train_percent)
        n_val_start = int(n_total * 0.9)
        
        # import pdb; pdb.set_trace()
        if args.dev:
            n_train = int(n_total * 0.0001)
            n_val_start = int(n_total * 0.9999)
            print(f"dev model, use less data. n_total={n_total},n_train={n_train},n_val_start={n_val_start}")

        if split == "train":
            self.frame_keys = self.frame_keys[:n_train]
        else:
            self.frame_keys = self.frame_keys[n_val_start:]

        self.groups = groups

        self.Mmax = 10

    def preload_all_feats(self):
        all_paths = []
        #import pdb; pdb.set_trace()
        for k in self.frame_keys:
            for r in self.groups[k]:
                all_paths.append(str(self.root_dir / r["slot_feat_path"]))

        self._feat_cache = {}
        for p in tqdm(all_paths, desc="preload feats"):
            self._feat_cache[p] = torch.load(p, map_location="cpu", weights_only=True).float().view(-1)
        print(f"load {len(self._feat_cache)} slot feats to cache")
        

    def __len__(self) -> int:
        return len(self.frame_keys)
    
    def _tokenize_one(self, cap:str) -> Tuple[List[int], int]:
        if cap in self._tok_cache:
            return self._tok_cache[cap]
        
        word_tokens = basic_tokenize(cap)
        words = word_tokens[: self.max_seq_len - 2]
        
        ids = [self.sos] + [self.vocab.get(w, self.unk) for w in words] + [self.eos]
        length = len(ids)
        
        if len(ids) < self.max_seq_len:
            ids = ids + [self.pad] * (self.max_seq_len - len(ids))
        else:
            ids = ids[: self.max_seq_len]
            length = min(length, self.max_seq_len)
        
        self._tok_cache[cap] = (ids, length)
        return ids, length
    
    def _sample_neg_index(self, idx: int) -> int:
        j = int(torch.randint(low=0, high=self._n - 1, size=(1,)).item())
        if j >= idx:
            j += 1
        return j
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.frame_keys[idx]
        recs = self.groups[key]
        
        img_path = self.root_dir / recs[0]["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.img_tf(image)
        
        whole_caption = "There is "
        gt_masks = []
        for r in recs:
            whole_caption = whole_caption + "a " + r["caption"] +", "
            gt_mask_path = self.root_dir / r["movi_mask_path"]
            mask_img = Image.open(gt_mask_path).convert("L")
            mask = self.img_tf(mask_img)  # 1, 128, 128
            mask = (mask > 0.5).float()
            gt_masks.append(mask)
        whole_caption = whole_caption + "in the image."
        
        caption_id, caption_len = self._tokenize_one(whole_caption)
        caption_id = torch.tensor(caption_id, dtype=torch.long)      # (L,)
        caption_len = torch.tensor(caption_len, dtype=torch.long)    # ()
        
        while len(gt_masks) < self.Mmax:
            gt_masks.append(torch.zeros_like(gt_masks[0]))
            
        out = {
            "video_idx": key[0],
            "frame_idx": key[1],
            "image": image,
            "tok_ids": caption_id,
            "tok_lns": caption_len,
            "gt_masks": torch.stack(gt_masks,dim=0)
        }
        
        if self.split == "val":
            rnd = random.Random(42+idx)
            
            neg_tok_ids = []
            neg_tok_lns = []
            
            for _ in range(3):
                j = rnd.randrange(len(self.frame_keys))
                if len(self.frame_keys) > 1 and j == idx:
                    j = (j + 1) % len(self.frame_keys)
                other_key = self.frame_keys[j]
                
                recs2 = self.groups[other_key]
                
                whole_caption2 = "There is "
                for r in recs2:
                    whole_caption2 = whole_caption2 + "a " + r["caption"] +", "
                whole_caption2 = whole_caption2 + "in the image."
                
                caption_id, caption_len = self._tokenize_one(whole_caption2)
                neg_tok_ids.append(torch.tensor(caption_id, dtype=torch.long))
                neg_tok_lns.append(torch.tensor(caption_len, dtype=torch.long))
            
            out["neg_tok_ids"] = torch.stack(neg_tok_ids, dim=0)       # List[Tensor(1,L)] len=3
            out["neg_tok_lns"] = torch.stack(neg_tok_lns, dim=0)     # List[Tensor(1,)]  len=3
        return out