import os
import hydra
#from typing import Optional
from ocl.cli import eval_utils
import hydra.core.global_hydra

import numpy as np
from ocl.datasets import _get_single_element_transforms

import timm


MOVIA_LABEL_ID2WORD = {
    "color_label": {
        0: "red",
        1: "blue",
        2: "green",
        3: "yellow",
        4: "orange",
        5: "purple",
        6: "pink",
        7: "brown",
    },
    "material_label": {
        0: "wood",
        1: "plastic",
    },
    "shape_label": {
        0: "cube",
        1: "circle",
        2: "square",
    },
    "size_label": {
        0: "small",
        1: "large",
    },
}

MOVIA_LABEL_PREFIX = ["color_label", "material_label", "shape_label", "size_label"]


def build_oclf_model_same_arch(
    train_config_path: str,
    checkpoint_path,
    overrides = None,
):
    """
    train_config_path: 
        outputs/.../<run>/.hydra/config.yaml
    checkpoint_path:
        outputs/.../<run>/checkpoints/epoch=...step=....ckpt
    overrides: 
    """
    train_config_path = hydra.utils.to_absolute_path(train_config_path)
    
    if train_config_path.endswith(".yaml"):
        config_dir, config_file = os.path.split(train_config_path)
        config_name = os.path.splitext(config_file)[0]
    else:
        config_dir, config_name = train_config_path, "config"
    
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = hydra.compose(config_name, overrides=overrides or [])

    datamodule, model = eval_utils.build_from_train_config(cfg, checkpoint_path)
    return datamodule, model



if __name__=="__main__":
    
    dm, lm = build_oclf_model_same_arch(
        "/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/outputs/slot_attention/movi_a/2026-02-02_05-55-06/config/config.yaml",
        "/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/outputs/slot_attention/movi_a/2026-02-02_05-55-06/lightning_logs/version_0/checkpoints/epoch=34-step=381993.ckpt",
    )
    
        # print(type(lm))          # <class 'ocl.combined_model.CombinedModel'>
        # print(lm.models.keys())  
        # print(lm.models['conditioning'])
        # print(lm.models['feature_extractor'])
        # print(lm.models['perceptual_grouping'])
        # print(lm.models['object_decoder'])
        # 'conditioning': RandomConditioning
        # 'feature_extractor': 
        # 'perceptual_grouping': 
        # 'object_decoder':
        
    if hasattr(lm.models["feature_extractor"], "module"):
        feat_extractor = lm.models["feature_extractor"].module
    else:
        feat_extractor = lm.models["feature_extractor"]
    backbone = feat_extractor.model
    backbone = backbone.module if hasattr(backbone, "module") else backbone
    src_sd = backbone.state_dict()
    
    vit = timm.create_model("vit_base_patch16_224_mocov3", pretrained=False)
    vit = vit.module if hasattr(vit, "module") else vit
    dst = vit
    
    missing, unexpected = dst.load_state_dict(src_sd, strict=False)
    
    import pdb; pdb.set_trace()
    
    dm.train_size = 16
    train_dataset = dm._create_webdataset(
        dm.train_shards,
        shuffle=dm.shuffle_train,
        n_datapoints=dm.train_size,
        keys_to_keep=("video", "instances", "segmentations"),
        transforms=_get_single_element_transforms(dm.train_transforms),
    )
    
    val_dataset = dm._create_webdataset(
        dm.val_shards,
        shuffle=dm.shuffle_train,
        n_datapoints=dm.val_size,
        keys_to_keep=("video", "instances", "segmentations"),
        transforms=_get_single_element_transforms(dm.train_transforms),
    )
    
    LABEL_FIELDS = ["color_label", "material_label", "shape_label", "size_label"]
      #color_label: 7
      #material_label: 1
      #shape_label: 2
      #size_label: 1
      
    from ocl.datasets import _get_batch_transforms
    train_loader = dm._create_dataloader(
        dataset=train_dataset,
        batch_transforms=_get_batch_transforms(dm.train_transforms),
        size=dm.train_size,
        batch_size=dm.batch_size,      
        partial_batches=False,
    )
    
    
    #for i,batch in enumerate(train_loader):
    #    import pdb; pdb.set_trace() 
    #    print(batch.keys())
    
    
    
    
    
    
    
    
    
    
    #sample = next(iter(dataset))
    #print(sample.keys())
    #print(type(sample["instances"]))
    #print(sample["instances"].keys())
    
        # metadata keys: ['backward_flow_range', 'depth_range', 'forward_flow_range', 'height', 'num_frames', 'num_instances', 'video_name', 'width']
        # instances keys: ['angular_velocities', 'bboxes_3d', 'color', 'color_label', 'friction', 'image_positions', 'mass', 'material_label', 'positions', 'quaternions', 'restitution', 'shape_label', 'size_label', 'velocities', 'visibility']
        # batch["image"].shape = torch.Size([8, 3, 128, 128])
        # batch['batch_size']=8
        

    #lm.to('cpu')
    #out = lm(batch)
        #out.keys() = dict_keys(['input', 'model', 'prefix', 'conditioning', 'feature_extractor', 'perceptual_grouping', 'object_decoder'])
        # 'input', 'model', 'prefix', 
        # out['conditioning'].shape = torch.Size([8, 11, 256])
        # 'feature_extractor', 
        # out['perceptual_grouping'].objects.shape = torch.Size([8, 11, 256])
        # out['object_decoder'].reconstruction.shape = torch.Size([8, 3, 128, 128])
        # out['object_decoder'].object_reconstructions.shape = torch.Size([8, 11, 3, 128, 128])
        # out['object_decoder'].masks.shape = torch.Size([8, 11, 128, 128])


    #feat = out["feature_extractor"]
        #print(type(feat))
        # print(feat.features.shape)     # (B, N, D)  ViT: D≈768；ResNet: D=C
        # print(feat.positions.shape)    # (N, 2)
        # print(getattr(feat, "aux_features", None))
    
    # backbone_feat = out["feature_extractor"].features
    # slot_feat = out['perceptual_grouping'].objects
    
    
    # loader = dm.train_dataloader()
    # split = "train"
    # loader = dm.train_dataloader()
    # batch = next(iter(loader))
    # print(f"[demo] batch keys = {list(batch.keys())}")
    # [demo] batch keys = ['__key__', 'image', 'batch_size']


