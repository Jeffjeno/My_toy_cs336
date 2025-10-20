import torch.nn as nn
import torch 
import math
from  einops import einsum
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device if device else None
        self.dtype = dtype if dtype else None
        self.gain = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        #===RMS
        rms = x.pow(2).mean(dim = -1 , keepdim= True).add(self.eps).sqrt()
        rms = 1/rms
        out = einsum(x,rms,"batch seq d_model, batch seq r -> batch seq d_model")
        out = einsum(out,self.gain,"b s d, d -> b s d")
        return out.to(in_type)