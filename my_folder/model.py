import json
import re


import torch
import torch.nn as nn
import torch.nn.functional as F

from my_utils import *
from utils.get_args import get_steve_args


class TransformerLayer(nn.Module):
    """
    A simple Transformer encoder block that supports:
    - self-attention when context is None
    - cross-attention style interface if needed later

    For the current use case, we only use self-attention on:
        [CLS] + 4 text-word embeddings + K slot embeddings
    """
    def __init__(self, query_dim: int, ff_dim: int, context_dim: int = None, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim if context_dim is not None else query_dim
        inner_dim = heads * dim_head

        self.norm1 = nn.LayerNorm(query_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(query_dim)
        self.ff = nn.Sequential(
            nn.Linear(query_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, query_dim),
        )

    def forward(
        self, 
        x: torch.Tensor, 
        key_padding_mask: torch.Tensor = None,
        return_attn: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [N, L, D]
            key_padding_mask: [N, L], True means "ignore this token"
            return_attn: whether to return attention weights
        Returns:
            [N, L, D]
        """
        h = self.norm1(x)
        attn_out, attn_weights = self.attn(
            h, h, h,
            # key_padding_mask=key_padding_mask,
            need_weights=return_attn,
        )
        
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.ff(h)
        if return_attn:
            return x, attn_weights
        return x

class SlotTextModel(nn.Module):
    def __init__(
        self, 
        args,
        oclf_model,
        textencoder: nn.Module,
        embed_dim = 256,
        slot_dim = 256,
    ):
        super().__init__()
        self.oclf_model = oclf_model
        for p in self.oclf_model.parameters():
            p.requires_grad_(False)

        self.text_encoder = textencoder
        self.embed_dim = embed_dim
        self.slot_dim = slot_dim

        self.use_vit = args.vit
        self.use_slot_select = getattr(args, "use_slot_select", False)
        self.pre_slot_feature = getattr(args, "pre_slot_feature", False)
        self.criterion = getattr(args, "criterion", "")
        self.tau = getattr(args, "tau", 0.07)
        self.steve = args.steve
        self.pre_slot_mask = getattr(args, "pre_slot_mask", False)
        self.vit_patches = getattr(args, "vit_patches", False)

        # Project slot features into the shared embedding dimension
        if self.pre_slot_feature or self.steve:
            self.vision_proj = nn.Linear(192, embed_dim)
        elif self.use_vit:
            self.vision_proj = nn.Linear(768, embed_dim)
        else:
            self.vision_proj = nn.Linear(slot_dim, embed_dim)

        # Project word embeddings into the shared embedding dimension
        self.text_proj = nn.Linear(getattr(textencoder, "embed_dim", embed_dim), embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        if self.use_vit and self.vit_patches:
            self.max_slots = 16
        elif self.steve:
            self.max_slots = 15
        else:
            self.max_slots = getattr(args, "num_slots", 11)
            
        self.max_text_len = 4
        
        # self.pos_emb = nn.Parameter(
        #     torch.randn(1, 1 + self.max_text_len + self.max_slots, embed_dim)
        # )

        self.text_pos_emb = nn.Parameter(
            torch.randn(1, 1 + self.max_text_len, embed_dim)   # CLS + text
        )

        self.vision_pos_emb = nn.Parameter(
            torch.randn(1, self.max_slots, embed_dim)          # only for vit side
        )
        
        self.tf_encoder = TransformerLayer(
            query_dim=embed_dim,
            ff_dim=embed_dim * 4,
            context_dim=embed_dim,
            heads=8,
            dim_head=64,
        )      
        
        # self.tf_layer = nn.TransformerEncoderLayer(
        #     d_model=embed_dim,
        #     nhead=8,
        #     dim_feedforward=256,
        #     dropout=0.1,
        #     activation="gelu",
        #     batch_first=True,
        #     norm_first=True,
        # )
        # self.tf_encoder = nn.TransformerEncoder(self.tf_layer, num_layers=1)

        self.match_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

        # self.match_head = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, 1),
        # )
        self.steve_model = None
        if args.steve:
            from steve.steve import STEVE
            from utils.get_args import get_steve_args
            self.steve_model = STEVE(get_steve_args())
            checkpoint_path ="/scratch/wz3008/checkpoints/steve_movi_a_2025-12-12T04:20:31.117982/checkpoint.pt.tar"
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            self.steve_model.load_state_dict(ckpt["model"], strict=True)

    @staticmethod
    def _l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)
    
    def _build_input_sequence(
        self,
        cls_tok: torch.Tensor,   # [B, 1, D]
        txt: torch.Tensor,       # [B, Ltxt, D]
        z_vis: torch.Tensor,     # [B, K, D]
    ) -> torch.Tensor:
        """
        Build transformer input:
        - CLS + text always get positional embeddings
        - visual tokens get pos emb only when self.use_vit == True
        """
        B, Ltxt, D = txt.shape
        K = z_vis.shape[1]

        txt_part = torch.cat([cls_tok, txt], dim=1)  # [B, 1+Ltxt, D]
        txt_part = txt_part + self.text_pos_emb[:, : 1 + Ltxt, :]

        if self.use_vit:
            vis_part = z_vis + self.vision_pos_emb[:, :K, :]
        else:
            vis_part = z_vis

        x = torch.cat([txt_part, vis_part], dim=1)   # [B, 1+Ltxt+K, D]
        return x
    
    def _run_oclf(self, input_dict: dict):
        """
        Run the frozen OCL model when slot features are not precomputed.
        """
        local_bs = input_dict["image"].shape[0]
        
        model_input = {
            "batch_size": local_bs,
            "image": input_dict["image"],
            "instances": None,
        }
        with torch.no_grad():
            self.oclf_model.eval()
            out = self.oclf_model(model_input)
        return out
    
    def _pool_vit_with_masks(self, vit_tok: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Pool ViT features inside object masks.

        Args:
            vit_tok: [B, N, 768], here N is expected to be 196 spatial patches
            masks:   [B, K, H, W] or [B, K, 1, H, W]

        Returns:
            pooled:  [B, K, 768]
        """
        if masks.dim() == 5:
            masks = masks.squeeze(2)  # [B, K, H, W]

        B, N, C = vit_tok.shape
        spatial_tok = vit_tok
        N_spatial = spatial_tok.shape[1]
        # (spatial_tok.shape)
        S = int(N_spatial ** 0.5)
        assert S * S == N_spatial, f"ViT spatial token count {N_spatial} is not a square number"

        feat_map = spatial_tok.transpose(1, 2).reshape(B, C, S, S)      # [B, 768, S, S]

        K = masks.shape[1]
        masks_f = F.interpolate(
            masks.float().view(B * K, 1, masks.shape[-2], masks.shape[-1]),
            size=(S, S),
            mode="nearest",
        ).view(B, K, 1, S, S) 

        masks_f = masks_f.flatten(3)                                     # [B, K, 1, S*S]
        feat_map_flat = feat_map.flatten(2).unsqueeze(1)                 # [B, 1, 768, S*S]
        pooled = (masks_f * feat_map_flat).sum(dim=-1) / masks_f.sum(dim=-1).clamp_min(1e-6)
        pooled = pooled.squeeze(2)                                       # [B, K, 768]
        return pooled

    def encode_text(self, word_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode word tokens into word-level embeddings.

        Args:
            word_tokens: [B, M, 4]

        Returns:
            t_emb: [B, M, 4, D]
        """
        t_emb = self.text_encoder(word_tokens)   # [B, M, 4, D_text]
        t_emb = self.text_proj(t_emb)            # [B, M, 4, D]
        return t_emb
    
    def encode_slots(self, input_dict: dict) -> torch.Tensor:
        """
        Encode image-side features into slot embeddings.

        Cases:
        1. pre_slot_feature=True: directly use input_dict["slot_feat"]
        2. use_vit=True: run OCL model, get ViT tokens, then pool them using slot masks
        3. otherwise: use OCL perceptual grouping slots directly

        Returns:
            z_slots: [B, K, D]
        """
        if self.pre_slot_feature:
            z_slots = input_dict["slot_feat"]                     # [B, K, D_pre]
            z_slots = self.vision_proj(z_slots)                  # [B, K, D]
            # z_slots = self._l2norm(z_slots)
            return z_slots
        
        if self.steve:
            with torch.no_grad():
                slots, _, _ = self.steve_model.encode(input_dict["image"].unsqueeze(1))
            z_slots = self.vision_proj(slots) 
            return z_slots.squeeze(1) # [B, Ks, D]
            
        out = self._run_oclf(input_dict)

        if self.use_vit:
            vit_tok = out["feature_extractor"].features          # [B, N, 768]
            
            if self.vit_patches:
                assert vit_tok.shape[1] == 196, f"Expected 196 ViT patches, got {vit_tok.shape}"
                B = vit_tok.shape[0]
                D2 = vit_tok.shape[2]
                x = vit_tok.view(B,14,14,D2)
                x = x.permute(0, 3, 1, 2)         # [B, 768, 14, 14]
                x = F.adaptive_avg_pool2d(x, (4, 4))  # [B, 768, 4, 4]
                x = x.permute(0, 2, 3, 1)         # [B, 4, 4, 768]
                x = x.reshape(B, 16, D2)           # [B, 16, 768]
                
                z_slots = self.vision_proj(x)       # [B, N, D]
                return z_slots
            if self.pre_slot_mask:
                masks = input_dict["pre_slot_mask"]              # [B, K, 1, H, W]
            else:
                masks = out["object_decoder"].masks                  # [B, K, H, W] or [B, K, 1, H, W]
            pooled = self._pool_vit_with_masks(vit_tok, masks)   # [B, K, 768]
            z_slots = self.vision_proj(pooled)                   # [B, K, D]
            # z_slots = self._l2norm(z_slots)
            return z_slots

        # Non-ViT slot branch
        slot_feat = out["perceptual_grouping"].objects       # [B, K, D_slot]
        z_slots = self.vision_proj(slot_feat)                    # [B, K, D]
        # z_slots = self._l2norm(z_slots)
        return z_slots
    
    def _build_seq_key_padding_mask(
        self,
        batch_size: int,
        text_len: int,
        slot_mask: torch.Tensor = None,
        repeat_n: int = None,
    ) -> torch.Tensor:
        """
        Build key_padding_mask for [CLS] + text + slots
        True = ignore this token
        """
        if slot_mask is None:
            return None

        # slot_mask: [K] or [N, K], where True means valid slot
        if slot_mask.dim() == 1:
            slot_mask = slot_mask.unsqueeze(0)

        if repeat_n is not None and slot_mask.shape[0] == 1:
            slot_mask = slot_mask.expand(repeat_n, -1)

        N, K = slot_mask.shape
        device = slot_mask.device

        cls_mask = torch.zeros(N, 1, dtype=torch.bool, device=device)       # keep CLS
        txt_mask = torch.zeros(N, text_len, dtype=torch.bool, device=device) # keep text
        slot_pad_mask = ~slot_mask.bool()                                    # True = ignore padded slot

        key_padding_mask = torch.cat([cls_mask, txt_mask, slot_pad_mask], dim=1)  # [N, 1+text_len+K]
        return key_padding_mask


    def multi_label_infonce_loss(
        self,
        t_emb: torch.Tensor,
        z_slots: torch.Tensor,
        caption_mask: torch.Tensor,
        slot_mask: torch.Tensor = None,
        tau: float = None,
    ):
        """
        Multi-positive / multi-label InfoNCE with loop-based computation.
    
        Args:
            t_emb:   [B, M, 4, D]
            z_slots: [B, K, D]
            caption_mask:  [B, M], True for real caption
            slot_mask:     [B, K], True for real slot
            tau:     temperature
        Returns:
            loss:       scalar
            loss_batch: [B]
            scores:     [B, B, M], where scores[i, j, r] = s_ij^(r)
        """
        if tau is None:
            tau = self.tau
    
        B, M, Ltxt, D = t_emb.shape
        B2, K, D2 = z_slots.shape
    
        assert B == B2, f"B mismatch: {t_emb.shape} vs {z_slots.shape}"
        assert D == D2, f"D mismatch: {t_emb.shape} vs {z_slots.shape}"
        assert caption_mask.shape == (B, M), f"caption_mask must be {(B, M)}, got {caption_mask.shape}"

    
        device = t_emb.device
        dtype = t_emb.dtype
    
        # scores[i, j, r] = score between text r from image i and slots from image j
        scores = torch.empty(B, B, M, device=device, dtype=dtype)
    
        for i in range(B):
            t_i = t_emb[i]                       # [M, 4, D]
            cap_mask_i = caption_mask[i].bool() # [M]
    
            for j in range(B):
                z_j = z_slots[j]                # [K, D]
    
                cls = self.cls_token.expand(M, 1, D)          # [M, 1, D]
                z_rep = z_j.unsqueeze(0).expand(M, K, D)      # [M, K, D]
                x = self._build_input_sequence(
                    cls_tok=cls,
                    txt=t_i,
                    z_vis=z_rep,
                )
                # x = torch.cat([cls, t_i, z_rep], dim=1)       # [M, 1+4+K, D]
                # x = x + self.pos_emb[:, : (1 + Ltxt + K), :]
    
                key_padding_mask = None
                if slot_mask is not None:
                    slot_mask_j = slot_mask[j].bool()          # [K]
                    key_padding_mask = self._build_seq_key_padding_mask(
                        batch_size=M,
                        text_len=Ltxt,
                        slot_mask=slot_mask_j,
                        repeat_n=M,
                    )                                           # [M, 1+4+K]
                
                h = self.tf_encoder(x, key_padding_mask=key_padding_mask)
                cls_out = h[:, 0, :]                          # [M, D]
                s_ij = self.match_head(cls_out).squeeze(-1)   # [M]
    
                # padded captions to -inf, not in logsumexp
                s_ij = s_ij.masked_fill(~cap_mask_i, float("-inf"))
                scores[i, j] = s_ij
    
        
        labels = torch.arange(B, device=device)

        logits_t2i = torch.logsumexp(scores / tau, dim=2)
        loss_t2i = F.cross_entropy(logits_t2i, labels, reduction="none")  # [B]

        logits_i2t = torch.logsumexp(scores / tau, dim=2).transpose(0, 1) # [B, B]
        loss_i2t = F.cross_entropy(logits_i2t, labels, reduction="none")  # [B]

        valid_per_sample = caption_mask.any(dim=1)  # [B]
        loss_batch = 0.5 * (loss_t2i + loss_i2t)

        if valid_per_sample.any():
            loss = loss_batch[valid_per_sample].mean()
        else:
            loss = loss_batch.mean()

        return loss, loss_batch, scores
    
    def four_way_eval_scores(
        self,
        word_tokens: torch.Tensor,
        z_slots: torch.Tensor,
        slot_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute 4-way matching scores for validation.

        Args:
            word_tokens: [4,4]
            z_slots:     [4, K, D]
            slot_mask:   [4, K]
        Returns:
            scores: [4]
        """
        B, K, D = z_slots.shape
        assert B == 4, f"Expected 4 candidates in validation, got {B}"

        word_tokens = word_tokens.unsqueeze(1)                    # [B, 1, 4]
        t_emb = self.encode_text(word_tokens)                     # [4, 1, 4, D]
        txt = t_emb[:, 0]                                       # [4, 4, D]
                
        cls_tok = self.cls_token.expand(B, 1, D)                 # [4, 1, D]
        # x = torch.cat([cls_tok, txt, z_slots], dim=1)                              # [4, 1+4+K, D]
        # x = x + self.pos_emb[:, :x.shape[1], :]                                # broadcast to [4, 1+4+K, D]
        x = self._build_input_sequence(
            cls_tok=cls_tok,
            txt=txt,
            z_vis=z_slots,
        )

        key_padding_mask = None
        if slot_mask is not None:
            key_padding_mask = self._build_seq_key_padding_mask(
                batch_size=B,
                text_len=txt.shape[1],
                slot_mask=slot_mask.bool(),
                repeat_n=None,
            )  # [4, 1+4+K]
            
        h = self.tf_encoder(x, key_padding_mask=key_padding_mask)              # [4, 1+4+K, D]
        scores = self.match_head(h[:, 0, :]).squeeze(-1)                       # [4]

        return scores

    def four_way_eval_scores_with_attn(
        self,
        word_tokens: torch.Tensor,
        z_slots: torch.Tensor,
        slot_mask: torch.Tensor = None,
    ):
        """
        Returns:
            scores: [4]
            cls_attn: [4, 1+4+K]   # each candidate's CLS attention to all tokens
        """
        B, K, D = z_slots.shape
        assert B == 4, f"Expected 4 candidates in validation, got {B}"

        word_tokens = word_tokens.unsqueeze(1)              # [4, 1, 4]
        t_emb = self.encode_text(word_tokens)               # [4, 1, 4, D]
        txt = t_emb[:, 0]                                   # [4, 4, D]

        cls_tok = self.cls_token.expand(B, 1, D)            # [4, 1, D]
        # x = torch.cat([cls_tok, txt, z_slots], dim=1)       # [4, 1+4+K, D]
        # x = x + self.pos_emb[:, :x.shape[1], :]
        x = self._build_input_sequence(
            cls_tok=cls_tok,
            txt=txt,
            z_vis=z_slots,
        )

        key_padding_mask = None
        if slot_mask is not None:
            key_padding_mask = self._build_seq_key_padding_mask(
                batch_size=B,
                text_len=txt.shape[1],
                slot_mask=slot_mask.bool(),
                repeat_n=None,
            )

        h, attn_weights = self.tf_encoder(
            x,
            key_padding_mask=key_padding_mask,
            return_attn=True,
        )   # h: [4, L, D], attn_weights: [4, L, L]

        scores = self.match_head(h[:, 0, :]).squeeze(-1)    # [4]
        cls_attn = attn_weights[:, 0, :]                    # [4, L]

        return scores, cls_attn
    
    def forward(self, input_dict: dict, mode: str = "train"):
        """
        Returns:
            Train: loss, loss_batch
            Val: scores [4]
        """
        word_tokens = input_dict["word_tokens"]

        if mode == "val_vis":
            z_slots = self.encode_slots(input_dict)
            slot_mask = input_dict.get("slot_mask", None)

            if self.use_vit and self.vit_patches:
                slot_mask = None

            scores, cls_attn = self.four_way_eval_scores_with_attn(
                word_tokens=word_tokens,
                z_slots=z_slots,
                slot_mask=slot_mask,
            )

            vis_masks = self.get_eval_visual_masks(input_dict)

            return {
                "scores": scores,         # [4]
                "cls_attn": cls_attn,     # [4, 1+4+K]
                "z_slots": z_slots,       # [4, K, D]
                "slot_mask": slot_mask,   # [4, K] or None
                "vis_masks": vis_masks,   # [4, K, 1, H, W] or None
            }
    
        if mode == "val":
            z_slots = self.encode_slots(input_dict)               # [4, K, D]   / B,N,D
            slot_mask = input_dict.get("slot_mask", None)
            
            if self.use_vit and self.vit_patches:
                slot_mask = None
                
            scores = self.four_way_eval_scores(
                word_tokens=word_tokens,
                z_slots=z_slots,
                slot_mask=slot_mask,
            )
            return scores
        
        caption_mask = input_dict["caption_mask"]
        slot_mask = input_dict.get("slot_mask", None)

        if self.use_vit and self.vit_patches:
            slot_mask = None
            
        t_emb = self.encode_text(word_tokens)           # [B, M, 4, D]
        z_slots = self.encode_slots(input_dict)         # [B, K, D]

        loss, _, _ = self.multi_label_infonce_loss(
            t_emb=t_emb,
            z_slots=z_slots,
            caption_mask=caption_mask,
            slot_mask=slot_mask,
        )
        return loss
    
    def visualize_steve(self, images, tau=0.1, hard=True):
        if self.steve_model is None:
            raise RuntimeError("STEVE is not enabled")
        with torch.no_grad():
            video = images.unsqueeze(1)  # [B, 1, C, H, W]
            recon_dvae, _, _, attns_vis = self.steve_model(video, tau=tau, hard=hard)
            recon_tf = self.steve_model.reconstruct_autoregressive(video)
        return video, recon_dvae, recon_tf, attns_vis

    def get_eval_visual_masks(self, input_dict: dict):
        """
        Returns:
            vis_masks: [B, K, 1, H, W] or None
        """
        # 1) if precomputed slot masks are available, use them directly
        if self.pre_slot_mask and "pre_slot_mask" in input_dict:
            vis_masks = input_dict["pre_slot_mask"].float()   # [B, K, 1, H, W]
            if vis_masks.dim() == 4:
                vis_masks = vis_masks.unsqueeze(2)
            return vis_masks

        # 2) STEVE branch: use STEVE attention visualization as soft slot masks
        if self.steve:
            with torch.no_grad():
                _, attns_vis, _ = self.steve_model.encode(input_dict["image"].unsqueeze(1))
                # attns_vis expected: [B, 1, K, C, H, W]
                if attns_vis.dim() == 6:
                    vis_masks = attns_vis[:, 0].mean(dim=2, keepdim=True)   # [B, K, 1, H, W]
                else:
                    vis_masks = None
            return vis_masks

        # 3) OCL branch: use object decoder masks
        out = self._run_oclf(input_dict)
        vis_masks = out["object_decoder"].masks
        # could be [B, K, H, W] or [B, K, 1, H, W]
        if vis_masks.dim() == 4:
            vis_masks = vis_masks.unsqueeze(2)
        return vis_masks.float()
                    
        
        



