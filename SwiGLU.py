import torch.nn as nn
import torch
import math
import einops
def SiLU( x):
    return x/(torch.exp(-x).add(1))

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff else 8/3*self.d_model
        std2= 2/(self.d_model+self.d_ff)
        std = math.sqrt(std2)
        self.W1 = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.d_ff,self.d_model),0,std,-3*std,3*std))
        self.W2 = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.d_model,self.d_ff),0,std,-3*std,3*std))
        self.W3 = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.d_ff,self.d_model),0,std,-3*std,3*std))
    def forward(self, x):
        out1 = einops.einsum(self.W1,x,"d_ff d_model, ... d_model -> ... d_ff")
        out1 = SiLU(out1)
        out2 = einops.einsum(self.W3,x,"d_ff d_model, ... d_model -> ... d_ff")
        out3 = einops.einsum(out1,out2,"... d_ff, ... d_ff -> ... d_ff")
        out = einops.einsum(self.W2,out3,"d_model d_ff , ... d_ff -> ... d_model")
        return out