import torch
import torch.nn as nn
import einops

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int , embedding_dim: int, device:torch.device | None = None, dtype:torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device if device else None
        self.dtype = dtype if dtype else None

        self.E = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.num_embeddings,self.embedding_dim),0,1,-3,3))
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        #token_ids:(batch_size, sequence_length)
        output = self.E[token_ids]
        return output