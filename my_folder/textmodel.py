import json

import torch
import torch.nn as nn



class TextEncoder(nn.Module):
    """
    Word-level text encoder for fixed-length 4-word captions.

    Input:
        word_tokens: [B, M, 4]
            B = batch size
            M = number of captions per image
            4 = number of words per caption

    Output:
        word_emb: [B, M, 4, D]
            D = embedding dimension
    """
    def __init__(self, embed_dim=256, json_path="/scratch/wz3008/new_SlotAttn/slot_attn_new/vocab.json"):
        super().__init__()
        self.embed_dim = embed_dim
        with open(json_path) as f:
            self.vocab = json.load(f)
        self.pad_id = self.vocab["<pad>"]
        self.vocab_size = len(self.vocab)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad_id)
    
    def forward(self, word_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            word_tokens: LongTensor of shape [B, M, 4]

        Returns:
            word_emb: FloatTensor of shape [B, M, 4, D]
        """
        return self.embedding(word_tokens)