import os
import hydra
#from typing import Optional
from ocl.cli import eval_utils
import hydra.core.global_hydra

def build_my_model_same_arch(
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
    dm, lm = build_my_model_same_arch(
        "/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/outputs/slot_attention/movi_a/2026-02-02_05-55-06/config/config.yaml",
        "/scratch/wz3008/new_SlotAttn/object-centric-learning-framework/outputs/slot_attention/movi_a/2026-02-02_05-55-06/lightning_logs/version_0/checkpoints/epoch=34-step=381993.ckpt",
        
    )
        #print(type(lm))          #<class 'ocl.combined_model.CombinedModel'>
        #print(lm.models.keys())  
        ##print(lm.models['conditioning'])
        #print(lm.models['feature_extractor'])
        #print(lm.models['perceptual_grouping'])
        #print(lm.models['object_decoder'])
        #'conditioning': RandomConditioning
        #'feature_extractor': 
        #'perceptual_grouping': 
        #'object_decoder':

    #import pdb; pdb.set_trace()
    batch = next(iter(dm.train_dataloader()))            
        # batch["image"].shape = torch.Size([8, 3, 128, 128])
        # batch['batch_size']=8
    
        #batch2 = {
        #  "input": {
        #    "image": batch["image"],
        #    "batch_size": batch.get("batch_size", batch["image"].shape[0]),
        #  }
        #}

    lm.to('cpu')
    out = lm(batch)
        #out.keys() = dict_keys(['input', 'model', 'prefix', 'conditioning', 'feature_extractor', 'perceptual_grouping', 'object_decoder'])
        # 'input', 'model', 'prefix', 
        # out['conditioning'].shape = torch.Size([8, 11, 256])
        # 'feature_extractor', 
        # out['perceptual_grouping'].objects.shape = torch.Size([8, 11, 256])
        # out['object_decoder'].reconstruction.shape = torch.Size([8, 3, 128, 128])
        # out['object_decoder'].object_reconstructions.shape = torch.Size([8, 11, 3, 128, 128])
        # out['object_decoder'].masks.shape = torch.Size([8, 11, 128, 128])

    #import pdb; pdb.set_trace()

    feat = out["feature_extractor"]
    print(type(feat))
    print(feat.features.shape)     # (B, N, D)  ViT: D≈768；ResNet: D=C
    print(feat.positions.shape)    # (N, 2)
    print(getattr(feat, "aux_features", None))
    
    backbone_feat = out["feature_extractor"].features
    slot_feat = out['perceptual_grouping'].objects



