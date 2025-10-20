import torch
import torch.nn as nn
import einops
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device if device else None 
        rule = lambda i,k : (i)/(theta ** ((2*k)/d_k))
        row = torch.arange(self.max_seq_len).unsqueeze(1)
        col = torch.arange(self.d_k//2).unsqueeze(0)
        value = rule(row,col)
        self.register_buffer("cosT",torch.cos(value),persistent= False)
        self.register_buffer("sinT",torch.sin(value),persistent= False)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        d_k_half = self.d_k//2
        out = einops.rearrange(x,"... s (half two) -> ... s half two",half = d_k_half,two = 2)
        # out = ... s d/22
        cos = self.cosT[token_positions] # s d/2
        sin = self.sinT[token_positions] # s d/2
        
        even = out[...,:,:,0]
        odd = out[...,:,:,1]
        even_rot = einops.einsum(even,cos,"... s half , ... s half -> ... s half ") - einops.einsum(odd,sin,"... s half , ... s half -> ... s half")
        odd_rot = einops.einsum(even,sin,"... s half , ... s half -> ... s half ") + einops.einsum(odd,cos,"... s half , ... s half -> ... s half")
        
        ret  = torch.stack([even_rot,odd_rot],dim= -1)
        ret = einops.rearrange(ret , " ... s h t -> ... s (h t)")
        return ret