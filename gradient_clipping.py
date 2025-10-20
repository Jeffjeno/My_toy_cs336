import torch
import math
def gradient_clipping( params , M):
    Total_norm = 0
    for p in params:
        if p.grad is None:
            continue
        Total_norm += (p.grad.data ** 2).sum()
        
    a =  math.sqrt(Total_norm)
    # return params if a < M else params * (M) / (a + 1e-6)
    if a < M:
        return params
    else:
        for p in params:
            if p.grad is None:
                continue 
            p.grad.data.mul_((M) / (a + 1e-6))
        return params