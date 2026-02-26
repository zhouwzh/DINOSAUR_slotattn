import os
import sys
from pathlib import Path

REPO_ROOT = "/scratch/wz3008/new_SlotAttn/object-centric-learning-framework"
sys.path.insert(0, REPO_ROOT)

from ocl.datasets import WebdatasetDataModule

print("SUCCESS")

os.environ["DATASET_PREFIX"] = "/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/scripts/datasets/outputs"

train_pattern = os.path.join(os.environ["DATASET_PREFIX"], "movi_a/train/shard-{000000..000328}.tar")

dm = WebdatasetDataModule(
    train_shards=train_pattern, 
    batch_size=1, 
    num_workers=0
)


it = dm._create_webdataset(
    uri_expression=train_pattern,
    shuffle=False,
    n_datapoints=None,
    keys_to_keep=("video", "segmentations"),  
    transforms=(),
)

sample = next(iter(it))
print("keys:", sample.keys())
print("__key__:", sample.get("__key__", None))
for k, v in sample.items():
    if k == "__key__":
        continue
    try:
        import torch
        if isinstance(v, torch.Tensor):
            print(k, "torch", tuple(v.shape), v.dtype)
        else:
            print(k, type(v))
    except Exception:
        print(k, type(v))