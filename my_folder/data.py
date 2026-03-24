# {"video_idx", "frame_idx", "instance_idx", "image_path", "movi_mask_path", "selected_slot_idx", "slot_feat_path", "slot_mask_path", "caption"}
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

VAL_METADATA_PATH = "/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/my_folder/val_metadata.json"

class SlotCaptionDataset(Dataset):
    def __init__(
        self,
        args,
        index_json: str,
        root_dir: str,
        split: str,
        train_percent: float = 0.9,
        val_metadata_path: str = VAL_METADATA_PATH,
    ):
        self.root_dir = Path(root_dir)
        self.index_path = Path(index_json)
        self.split = split
        self.train_percent = train_percent
        val_metadata_path = args.val_metadata_path
        print(f"using val_metadata_path: {args.val_metadata_path}")
        self.val_metadata_path = Path(val_metadata_path)

        with open("/scratch/wz3008/new_SlotAttn/slot_attn_new/vocab.json") as f:
            self.vocab = json.load(f)

        self.sos = self.vocab["<sos>"]
        self.eos = self.vocab["<eos>"]
        self.pad = self.vocab["<pad>"]
        self.unk = self.vocab["<unk>"]
        self.max_seq_len = 25
        self.word_seq_len = 4   # default 4 words for each capton
        self.Mmax = 2

        self._tok_cache: Dict[str, Tuple[List[int], int]] = {}
        self._word_tok_cache: Dict[str, List[int]] = {}

        self.image_size = args.image_size
        self.img_tf = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
        ])

                            
        self.pre_slot_feature = args.pre_slot_feature
        self.pre_slot_mask = args.pre_slot_mask

         # --------------------------------------------------
        # load groups from index_json
        # groups[(video_idx, frame_idx)] = [rec1, rec2, ...]
        # --------------------------------------------------
        groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (int(rec["video_idx"]), int(rec["frame_idx"]))
                groups.setdefault(key, []).append(rec)

        self.groups = groups
        self.all_frame_keys = sorted(groups.keys())

        n_total = len(self.all_frame_keys)
        n_train = int(n_total * self.train_percent)
        n_val_start = int(n_total * 0.9)

        if args.dev:
            n_train = int(n_total * 0.01)
            n_val_start = int(n_total * 0.99)
            print(f"dev mode, use less data. n_total={n_total}, n_train={n_train}, n_val_start={n_val_start}")

        raw_train_frame_keys = self.all_frame_keys[:n_train]
        self.val_frame_keys = self.all_frame_keys[n_val_start:]

        if self.split == "train":
            self.train_frame_keys = self._build_interleaved_train_keys(raw_train_frame_keys, seed=args.seed)
            self.frame_keys = self.train_frame_keys
        elif self.split == "val":
            self.valdata_preprocess()
            with open(self.val_metadata_path, "r", encoding="utf-8") as f:
                self.val_metadata = json.load(f)
        else:
            raise ValueError(f"split must be 'train' or 'val', got {self.split}")

    def _build_interleaved_train_keys(self, frame_keys, seed=42):
        """
        train frame keys, group in video, interleaved arrange
        Let adjacent samples come from different videos.
        """
        rng = random.Random(seed)

        # video_idx -> list[(video_idx, frame_idx)]
        video_to_keys = {}
        for key in frame_keys:
            video_idx = int(key[0])
            video_to_keys.setdefault(video_idx, []).append(key)

        # 每个视频内部打乱
        for video_idx in video_to_keys:
            rng.shuffle(video_to_keys[video_idx])

        # 视频顺序打乱
        video_ids = list(video_to_keys.keys())
        rng.shuffle(video_ids)

        # round-robin 
        interleaved = []
        still_has_data = True
        while still_has_data:
            still_has_data = False
            for video_idx in video_ids:
                if len(video_to_keys[video_idx]) > 0:
                    interleaved.append(video_to_keys[video_idx].pop())
                    still_has_data = True

        return interleaved
    # ======================================================
    # tokenization
    # ======================================================
    def _tokenize_one(self, cap: str) -> Tuple[List[int], int]:
        if cap in self._tok_cache:
            return self._tok_cache[cap]

        words = basic_tokenize(cap)[: self.max_seq_len - 2]
        ids = [self.sos] + [self.vocab.get(w, self.unk) for w in words] + [self.eos]
        length = len(ids)
        if len(ids) < self.max_seq_len:
            ids = ids + [self.pad] * (self.max_seq_len - len(ids))
        else:
            ids = ids[: self.max_seq_len]
            length = min(length, self.max_seq_len)
        self._tok_cache[cap] = (ids, length)
        return ids, length

    def _word_tokenize_one(self, cap: str) -> List[int]:
        """
        output shape [4]
        """
        if cap in self._word_tok_cache:
            return self._word_tok_cache[cap]

        words = basic_tokenize(cap)[: self.word_seq_len]
        ids = [self.vocab.get(w, self.unk) for w in words]

        if len(ids) < self.word_seq_len:
            ids = ids + [self.pad] * (self.word_seq_len - len(ids))

        self._word_tok_cache[cap] = ids
        return ids

    def get_image(self, img_path: Path):
        image = Image.open(img_path).convert("RGB")
        image = self.img_tf(image)
        return image
    
    def get_greyscale_image(self, img_path: Path):
        image = Image.open(img_path).convert("L")
        image = self.img_tf(image)
        return image

    def get_slot_feature(self, feat_path: str):
        if hasattr(self, "_feat_cache") and feat_path in self._feat_cache:
            feat = self._feat_cache[feat_path]
        else:
            feat = torch.load(feat_path, map_location="cpu", weights_only=True).float().view(-1)
        return feat

    def _build_frame_slot_feats(self, recs: List[Dict[str, Any]]) -> torch.Tensor:
        """
        output: [Mmax, D]
        """
        if not self.pre_slot_feature:
            return torch.zeros(0)

        feats = []
        for r in recs:
            p = str(self.root_dir / r["slot_feat_path"])
            feat = self.get_slot_feature(p)
            feats.append(feat)

        if len(feats) == 0:
            raise ValueError("empty feats in _build_frame_slot_feats")

        slot_dim = feats[0].shape[-1]
        valid_n = min(len(feats), self.Mmax)

        while len(feats) < self.Mmax:
            feats.append(torch.zeros(slot_dim))

        feats = feats[: self.Mmax]
        return torch.stack(feats, dim=0)   # [Mmax, D]

    def _build_frame_slot_mask(self, recs: List[Dict[str, Any]]) -> torch.Tensor:
        """
        output: [Mmax], 1 for valid slot feat, 0 for padded slot feat
        """
        valid_n = min(len(recs), self.Mmax)
        mask = torch.zeros(self.Mmax, dtype=torch.bool)
        mask[:valid_n] = True
        return mask

    def preload_all_feats(self):
        all_paths = []
        if self.split == "train":
            keys_to_use = self.frame_keys
        else:
            keys_to_use = self.val_frame_keys

        for k in keys_to_use:
            for r in self.groups[k]:
                all_paths.append(str(self.root_dir / r["slot_feat_path"]))

        self._feat_cache = {}
        for p in tqdm(all_paths, desc="preload feats"):
            self._feat_cache[p] = torch.load(p, map_location="cpu", weights_only=True).float().view(-1)

        print(f"load {len(self._feat_cache)} slot feats to cache")
    
    # ======================================================
    # val metadata preprocess / check

    # ======================================================
    def valdata_preprocess(self):
        if self.val_metadata_path.exists():
            print(f"val metadata exists, skip: {self.val_metadata_path}")
            return

        print(f"building val metadata: {self.val_metadata_path}")

        rng = random.Random(42)

        # frame -> set(captions in this frame)
        frame_caption_sets: Dict[Tuple[int, int], set] = {}
        for k in self.val_frame_keys:
            frame_caption_sets[k] = set(r["caption"] for r in self.groups[k])

        val_items = []

        # construct a 4-way pair for every object caption in val frame
        for pos_key in tqdm(self.val_frame_keys, desc="build val metadata"):
            recs = self.groups[pos_key]
            for r in recs:
                target_label = r["caption"]

                negative_pool = []
                for other_key in self.val_frame_keys:
                    if other_key == pos_key:
                        continue
                    if target_label not in frame_caption_sets[other_key]:
                        negative_pool.append(other_key)

                if len(negative_pool) < 3:
                    continue

                neg_keys = rng.sample(negative_pool, 3)

                image_keys = [
                    {
                        "video_idx": int(pos_key[0]),
                        "frame_idx": int(pos_key[1]),
                        "is_positive": 1,
                    }
                ]
                for nk in neg_keys:
                    image_keys.append(
                        {
                            "video_idx": int(nk[0]),
                            "frame_idx": int(nk[1]),
                            "is_positive": 0,
                        }
                    )

                rng.shuffle(image_keys)
                target_index = next(i for i, x in enumerate(image_keys) if x["is_positive"] == 1)

                val_items.append(
                    {
                        "target_label": target_label,
                        "target_index": target_index,
                        "positive_key": {
                            "video_idx": int(pos_key[0]),
                            "frame_idx": int(pos_key[1]),
                        },
                        "image_keys": image_keys,
                    }
                )

        self.val_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.val_metadata_path, "w", encoding="utf-8") as f:
            json.dump(val_items, f, indent=2)

        print(f"saved val metadata to: {self.val_metadata_path}")
        print(f"num val pairs: {len(val_items)}")

    def check_val_metadata(self, verbose: bool = True) -> bool:
        if not self.val_metadata_path.exists():
            print(f"val metadata not found: {self.val_metadata_path}")
            return False

        with open(self.val_metadata_path, "r", encoding="utf-8") as f:
            val_metadata = json.load(f)

        num_total = 0
        num_ok = 0
        num_bad = 0

        for i, item in enumerate(val_metadata):
            ok = True
            errors = []

            target_label = item["target_label"]
            image_keys = item["image_keys"]

            if len(image_keys) != 4:
                ok = False
                errors.append(f"expected 4 image_keys, got {len(image_keys)}")

            keys = []
            pos_count = 0

            for j, kinfo in enumerate(image_keys):
                key = (int(kinfo["video_idx"]), int(kinfo["frame_idx"]))
                keys.append(key)

                if key not in self.groups:
                    ok = False
                    errors.append(f"image_keys[{j}] not found in groups: {key}")
                    continue

                recs = self.groups[key]
                cap_set = set(r["caption"] for r in recs)
                contains = target_label in cap_set

                if contains:
                    pos_count += 1

                if int(kinfo["is_positive"]) == 1 and not contains:
                    ok = False
                    errors.append(f"image_keys[{j}] marked positive but does not contain target_label")
                if int(kinfo["is_positive"]) == 0 and contains:
                    ok = False
                    errors.append(f"image_keys[{j}] marked negative but actually contains target_label")

            if len(keys) != len(set(keys)):
                ok = False
                errors.append("duplicate image keys in one item")

            if pos_count != 1:
                ok = False
                errors.append(f"expected exactly 1 image containing target_label, got {pos_count}")

            if "target_index" in item:
                ti = int(item["target_index"])
                if ti < 0 or ti >= len(image_keys):
                    ok = False
                    errors.append(f"target_index out of range: {ti}")
                elif int(image_keys[ti]["is_positive"]) != 1:
                    ok = False
                    errors.append("target_index does not point to the positive image")

            num_total += 1
            if ok:
                num_ok += 1
            else:
                num_bad += 1
                if verbose:
                    print(f"\n[ERROR] item {i}")
                    for e in errors:
                        print(" -", e)

        print("\n===== VAL METADATA CHECK SUMMARY =====")
        print(f"total: {num_total}")
        print(f"ok: {num_ok}")
        print(f"bad: {num_bad}")

        return num_bad == 0

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.frame_keys)
        else:
            return len(self.val_metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.split == "train":
            return self._getitem_train(idx)
        else:
            return self._getitem_val(idx)
    
    # ======================================================
    # train getitem

    # ======================================================
    def _getitem_train(self, idx: int) -> Dict[str, Any]:
        key = self.frame_keys[idx]
        recs = self.groups[key]

        img_path = self.root_dir / recs[0]["image_path"]
        image = self.get_image(img_path)

        token_ids_list: List[List[int]] = []
        token_len_list: List[int] = []
        word_tokens_list: List[List[int]] = []
        gt_masks = []
        feats = []
        pre_slot_masks = []

        valid_n = min(len(recs), self.Mmax)

        # whole_caption = "There is "
        for r in recs[: self.Mmax]:
            cap = r["caption"]

            # ids, ln = self._tokenize_one(cap)
            # token_ids_list.append(ids)
            # token_len_list.append(ln)

            word_tok = self._word_tokenize_one(cap)
            word_tokens_list.append(word_tok)

            # whole_caption = whole_caption + "a " + cap + ", "

            gt_mask_path = self.root_dir / r["movi_mask_path"]
            mask_img = Image.open(gt_mask_path).convert("L")
            mask = self.img_tf(mask_img)
            mask = (mask > 0.5).float()
            gt_masks.append(mask)

            if self.pre_slot_feature:
                p = str(self.root_dir / r["slot_feat_path"])
                feat = self.get_slot_feature(p)
                feats.append(feat)

            if self.pre_slot_mask:
                p = str(self.root_dir / r["slot_mask_path"])
                pre_slot_mask = self.get_greyscale_image(p)   # [1, H, W], binary mask where 1 means this slot is valid, 0 means this slot is padded
                pre_slot_masks.append(pre_slot_mask)


        caption_mask = torch.zeros(self.Mmax, dtype=torch.bool)
        caption_mask[:valid_n] = True
        # whole_caption = whole_caption + "in the image."

        # caption_id, caption_len = self._tokenize_one(whole_caption)
        slot_mask = torch.zeros(self.Mmax, dtype=torch.bool)
        # caption_id = torch.tensor(caption_id, dtype=torch.long)
        slot_mask[:valid_n] = True
        # caption_len = torch.tensor(caption_len, dtype=torch.long)

        if self.pre_slot_feature:
            if len(feats) == 0:
                raise ValueError(f"no slot feats found for key={key}")
            slot_dim = feats[0].shape[-1]
        
        if self.pre_slot_mask:
            if len(pre_slot_masks) == 0:
                raise ValueError(f"no pre slot masks found for key={key}")

        while len(gt_masks) < self.Mmax:
            gt_masks.append(torch.zeros_like(gt_masks[0]))
            # token_ids_list.append([self.pad] * self.max_seq_len)
            # token_len_list.append(1)
            word_tokens_list.append([self.pad] * self.word_seq_len)

            if self.pre_slot_feature:
                feats.append(torch.zeros(slot_dim))
            if self.pre_slot_mask:
                pre_slot_masks.append(torch.zeros_like(pre_slot_masks[0]))

        gt_masks = gt_masks[: self.Mmax]
        # token_ids_list = token_ids_list[: self.Mmax]
        # token_len_list = token_len_list[: self.Mmax]
        word_tokens_list = word_tokens_list[: self.Mmax]

        if self.pre_slot_feature:
            feats = feats[: self.Mmax]
            slot_feats = torch.stack(feats, dim=0)   # [Mmax, D]
        else:
            slot_feats = torch.zeros(0)
        
        if self.pre_slot_mask:
            pre_slot_masks = pre_slot_masks[: self.Mmax]
            pre_slot_masks = torch.stack(pre_slot_masks, dim=0)   # [Mmax, C, H, W]
        else:
            pre_slot_masks = torch.zeros(0)

        out = {
            "video_idx": key[0],
            "frame_idx": key[1],
            "image": image,
            "gt_masks": torch.stack(gt_masks, dim=0),                     # [Mmax, 1, H, W]
            "word_tokens": torch.tensor(word_tokens_list, dtype=torch.long),  # [Mmax, 4]
            "caption_mask": caption_mask,                                 # [Mmax]
            "slot_mask": slot_mask,                                       # [Mmax]
            "slot_feat": slot_feats,
            "pre_slot_mask": pre_slot_masks,
            # "tok_ids": caption_id,
            # "tok_lns": caption_len,
            # "word_ids": torch.tensor(token_ids_list, dtype=torch.long),   # [Mmax, L]
            # "word_lens": torch.tensor(token_len_list, dtype=torch.long),  # [Mmax]
        }
        return out

    def _getitem_val(self, idx: int) -> Dict[str, Any]:
        item = self.val_metadata[idx]

        target_label = item["target_label"]
        target_index = int(item["target_index"])

        # tok_ids, tok_lns = self._tokenize_one(target_label)
        word_tokens = self._word_tokenize_one(target_label)

        images = []
        image_keys = []
        contains_target = []
        slot_feats_all = []
        slot_masks_all = []
        pre_slot_masks_all = []

        for kinfo in item["image_keys"]:
            key = (int(kinfo["video_idx"]), int(kinfo["frame_idx"]))
            recs = self.groups[key]

            img_path = self.root_dir / recs[0]["image_path"]
            images.append(self.get_image(img_path))
            image_keys.append([key[0], key[1]])
            contains_target.append(int(kinfo["is_positive"]))

            if self.pre_slot_feature:
                slot_feats_all.append(self._build_frame_slot_feats(recs))
                slot_masks_all.append(self._build_frame_slot_mask(recs))
            
            if self.pre_slot_mask:
                cur_pre_slot_masks = []
                valid_n = min(len(recs), self.Mmax)
                for r in recs[: self.Mmax]:
                    p = str(self.root_dir / r["slot_mask_path"])
                    pre_slot_mask = self.get_greyscale_image(p)   # [1, H, W], binary mask where 1 means this slot is valid, 0 means this slot is padded
                    cur_pre_slot_masks.append(pre_slot_mask)
            
                if len(cur_pre_slot_masks) == 0:
                    raise ValueError(f"no pre slot masks found for key={key}")
                
                while len(cur_pre_slot_masks) < self.Mmax:
                    cur_pre_slot_masks.append(torch.zeros_like(cur_pre_slot_masks[0]))

                cur_pre_slot_masks = cur_pre_slot_masks[: self.Mmax]
                cur_pre_slot_masks = torch.stack(cur_pre_slot_masks, dim=0)   # [Mmax, C, H, W]
                pre_slot_masks_all.append(cur_pre_slot_masks)

        if self.pre_slot_feature:
            slot_feats = torch.stack(slot_feats_all, dim=0)   # [4, Mmax, D]
            slot_masks = torch.stack(slot_masks_all, dim=0)   # [4, Mmax]
        else:
            slot_feats = torch.zeros(4, 0)
            slot_masks = torch.zeros(4, 0, dtype=torch.bool)
        if self.pre_slot_mask:
            pre_slot_masks = torch.stack(pre_slot_masks_all, dim=0)   # [4, Mmax, C, H, W]
        else:
            pre_slot_masks = torch.zeros(4, 0)

        out = {
            # "tok_ids": torch.tensor(tok_ids, dtype=torch.long),            # [L]
            # "tok_lns": torch.tensor(tok_lns, dtype=torch.long),            # []
            "word_tokens": torch.tensor(word_tokens, dtype=torch.long),    # [4]
            "target_label": target_label,
            "target_index": torch.tensor(target_index, dtype=torch.long),
            "images": torch.stack(images, dim=0),                          # [4, C, H, W]
            # "image_keys": torch.tensor(image_keys, dtype=torch.long),      # [4, 2]
            "contains_target": torch.tensor(contains_target, dtype=torch.long),
            "slot_feats": slot_feats,
            "slot_masks": slot_masks,
            "pre_slot_mask": pre_slot_masks,
        }
        return out
