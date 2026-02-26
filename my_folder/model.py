import json
import re


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence

from my_utils import *

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
        self.vision_proj = nn.Linear(slot_dim, embed_dim)
        self.text_proj = nn.Linear(getattr(textencoder, "embed_dim", embed_dim), embed_dim)
        
        self.use_vit = args.vit
        if self.use_vit:
            self.vision_proj = nn.Linear(768, embed_dim)
            
        self.use_slot_select = args.use_slot_select
    
    @staticmethod
    def _l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)
        
    def forward(self, input):
        # B = input['batch_size']
        tok_ids = input['instances']['tok_ids']  # B, L
        tok_lns = input['instances']['tok_lns']  # B
        # image = input['image']  # [8, 3, 224, 224]
        # segmentations = input['segmentations']   # B, 24, 128, 128, 1
        
        input['image'] = input['image'].cuda()
        
        
        # backbone_feat = out["feature_extractor"].features
        with torch.no_grad():
            out = self.oclf_model(input)
            
        if self.use_slot_select:
            slot_feat, slot_masks = select_slot(
                out['perceptual_grouping'].objects, 
                out['object_decoder'].masks,
                input['gt_masks'],           # B, 10, 1, 128, 128
            )
        else:
            slot_feat = out['perceptual_grouping'].objects
            slot_masks = out['object_decoder'].masks
        
        if not self.use_vit:
            # slot_feat = out['perceptual_grouping'].objects  #[8, 11, 256]
            z_slots = self._l2norm(self.vision_proj(slot_feat))
        else:
            vit_tok = out["feature_extractor"].features   # B, 197, 768
            # masks = out['object_decoder'].masks           # B, 11, 128, 128
            masks = slot_masks
            B,N,C = vit_tok.shape
            S = int(N ** 0.5)
            feat_map = vit_tok.transpose(1,2).reshape(B,C,S,S)
            masks_f = F.interpolate(
                masks.float().view(B*masks.shape[1], 1, masks.shape[-2], masks.shape[-1]),
                size = (S, S),
                mode = "nearest"
            ).view(B, masks.shape[1], 1, S, S) #B 11 1 S S
            masks_f = masks_f.flatten(3) # (B,M,1,196)
            feat_map_flat = feat_map.flatten(2).unsqueeze(1)   # B, 1, 768, 768
            pooled = (masks_f * feat_map_flat).sum(dim=-1) / masks_f.sum(dim=-1).clamp_min(1e-6)
            z_slots = self._l2norm(self.vision_proj(pooled))
        
        
        t_slots = self.text_encoder.forward_ids(tok_ids,tok_lns)  #B,Dtxt
        t_slots = self._l2norm(self.text_proj(t_slots)) #B,256
            
        t_scene = t_slots
        z_scene = z_slots.sum(dim=1)
        
        loss = self.infonce_loss(z_scene, t_scene)
        
        return loss.mean(), loss
        
    def infonce_loss(self, z, t):
        B = z.shape[0]
        tau = getattr(self, "tau", 0.07)
        logits = (z @ t.t()) / tau
        labels = torch.arange(B, device=logits.device)
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_i2t + loss_t2i)
        
        loss_batch = F.cross_entropy(logits, labels, reduction="none")
        return loss_batch
    

class TextEncoder(nn.Module):
    """
    - Python hash: token to [0, vocab_size)
    - nn.Embedding + GRU encoder
    - change to  CLIP / LLaMA encoder。
    """
    """
    - tokenizer:
    -vocab: 
    -special token:
    -padding mask:
    -nlp model: need to match nlp model's input and output
    """
    def __init__(self, embed_dim: int = 256, vocab_size: int = 10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        
        # #self.nlp = spacy.load("en_core_web_sm")
        
        
        with open("/scratch/wz3008/new_SlotAttn/slot_attn_new/vocab.json") as f:
            self.vocab = json.load(f)
            
        self.pad_id = 1
    
    
    def forward_ids(self, token_ids: torch.Tensor, lengths: torch.Tensor)-> torch.Tensor:
        """
        token_ids: (N, L)  long
        lengths:   (N,)    long
        return: sent_emb (N, D)
        """
        lengths = lengths.clamp(min=1).cpu()
        emb = self.embedding(token_ids)  # (N,L,D)
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)        # (1,N,D)
        return h_n.squeeze(0)            # (N,D)
    
    @staticmethod
    def _basic_tokenize(text: str):
        return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z\d]", text.lower())
    
    def tokenize(self, captions):
        """
        captions: List[str]
        token_ids: List[List[int]], len=N, each is the list of caption's token id
        """
        max_seq_len = 25
        all_tokens = []
        token_lengths = []
        # if isinstance(captions, str):
        #     captions = [captions]

        for cap in captions:
            #doc = self.nlp(cap)
            #word_tokens = [token.text for token in doc]
            word_tokens = self._basic_tokenize(cap)

            if len(word_tokens) > max_seq_len - 2:
                word_tokens = word_tokens[:max_seq_len - 2]
            
            token_lenght = len(word_tokens) + 2

            tokens = [self.vocab["<sos>"]] + [self.vocab.get(token, self.vocab["<unk>"]) for token in word_tokens] + [self.vocab["<eos>"]] + [self.vocab["<pad>"]] * (max_seq_len - len(word_tokens) - 2)
            all_tokens.append(tokens)
            token_lengths.append(token_lenght)
        return all_tokens, token_lengths

    def forward(self, captions)-> torch.Tensor:
        """
        captions: List[str]
        sent_emb: (N, D): Tensor
        """
        token_ids, token_lengths = self.tokenize(captions)  
        max_len = max(len(ids) for ids in token_ids)

        # padding
        padded = []
        lengths = []
        for ids in token_ids:
            lengths.append(len(ids))
            if len(ids) < max_len:
                ids = ids + [0] * (max_len - len(ids))
            padded.append(ids)

        padded = torch.tensor(padded, dtype=torch.long, device=self.embedding.weight.device)  # (N, L)
        emb = self.embedding(padded)  # (N, L, D)

        # GRU
        out, h_n = self.gru(emb)      # h_n: (1, N, D)
        sent_emb = h_n.squeeze(0)     # (N, D)
        return sent_emb
        



        
        