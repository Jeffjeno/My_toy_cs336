import torch
from cs336_basics.softmax import softmax
from einops import einsum
import math
def scaled_dot_product_attention( Q ,K ,V , mask = None):
    d_k_sqrt = math.sqrt(Q.shape[-1])
    
    Pre = einsum(Q,K,"b ... querys d , b ... keys d -> b ... querys keys")/d_k_sqrt
    
    if mask is not None:
        Pre = Pre.masked_fill(mask==0,-torch.inf)
    
    Pre = softmax(Pre,-1)
    
    return einsum(Pre,V,"b ... querys keys , b ... keys d_v -> b ... querys d_v")
    