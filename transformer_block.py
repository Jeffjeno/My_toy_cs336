import torch 
import torch.nn as nn
from cs336_basics.multihead_self_attention import Multi_Head_Self_attention
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.linear import Linear
from cs336_basics.SwiGLU import SwiGLU

class Transformer_Block(nn.Module):
    def __init__(self, d_model:int ,num_heads:int, d_ff : int , max_seq_len : int | None = None ,theta: float | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        if max_seq_len and theta:
            self.max_seq_len = max_seq_len
            self.theta = theta
        self.MHA = Multi_Head_Self_attention(self.d_model,
                                             self.num_heads ,
                                             self.max_seq_len if max_seq_len is not None else None ,
                                             self.theta if theta is not None else None)

        self.rms1 = RMSNorm(self.d_model)

        self.ffn = SwiGLU(self.d_model,self.d_ff)
        self.rms2 = RMSNorm(self.d_model)
        
    def forward(self , x : torch.Tensor): 
        """_summary_

        Args:
            x (torch.Tensor): (batch_size, seq_len, d_model)
        """
        B,S,_ = x.shape
        token_positions = torch.arange(S)
        out1 = x + self.MHA.forward(self.rms1.forward(x), token_positions)
        out2 = out1 + self.ffn.forward(self.rms2.forward(out1))
        return out2