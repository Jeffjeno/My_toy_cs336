import torch.nn as nn
import torch
from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import Transformer_Block
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.linear import Linear
from cs336_basics.softmax import softmax
class Transformer_LM(nn.Module):
    def __init__(self,vocab_size: int,context_length: int ,num_layers: int ,
    d_model: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff =d_ff
        self.d_model =d_model
        self.rope_theta = rope_theta
        self.layers = nn.ModuleList([Transformer_Block(self.d_model,self.num_heads,self.d_ff,self.context_size,self.rope_theta) for _ in range(self.num_layers)])
        self.rms = RMSNorm(self.d_model)
        self.embedding =Embedding(self.vocab_size,self.d_model)
        self.linear = Linear(self.d_model,self.vocab_size)
    def forward(self, x):
        #Int[Tensor, "batch_size sequence_length"]
        x = self.embedding.forward(x)
        #(B, s, d_model)
        for i in range(self.num_layers):
            x = self.layers[i].forward(x)
        #batch sequence_length d_model
        x = self.rms.forward(x)
        
        x = self.linear.forward(x)
        # vocab_size d_model ,batch sequence_length d_model : num: 2*batch sequence_length* vocab_size * d_model
        return x