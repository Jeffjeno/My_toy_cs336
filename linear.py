import torch.nn as nn
import torch
import math
from einops import rearrange, einsum
class Linear(nn.Module):
    def __init__(self,in_features: int, out_features: int , device: torch.device | None =None, dtype: torch.dtype| None =None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device if device else None
        self.dtype = dtype if dtype else None
        std2= 2/(self.in_features+self.out_features)
        std = math.sqrt(std2)
        self.W = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.out_features,self.in_features),0,std,-3*std,3*std))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = einsum(self.W,x,"out In, ... In -> ... out")
        return out