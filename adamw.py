import torch 
import torch.nn as nn
import math
from  collections.abc import Callable, Iterable
from typing import Optional


class AdamW(torch.optim.Optimizer):
    def __init__(self, params,weight_decay, lr,betas,eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"alpha": lr,
                    "beta1": betas[0],
                    "beta2": betas[1],
                    "eps" : eps,
                    "lamda": weight_decay
                    }
        
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            alpha = group["alpha"] # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lamda = group["lamda"]
            eps = group["eps"]
            for p in group["params"]:
                m = self.state[p].get("m",torch.zeros_like(p.data))
                v = self.state[p].get("v",torch.zeros_like(p.data)) 
                ###
                if p.grad is None:
                    continue 
                state = self.state[p] # Get sta
                t = state.get("t", 1)   
                grad = p.grad.data 
                m= grad*(1-beta1) + beta1*m
                v= (1-beta2) * ( grad ** 2) + beta2*v
                alpha_t = alpha * (math.sqrt(1- beta2 ** t ))/ ( 1 - beta1 ** t)
                p.data -= alpha_t * m / ( v.sqrt().add(eps)) 
                p.data -= alpha * lamda * p.data    
                state["t"] = t + 1 
                #####
                self.state[p]["m"]= m
                self.state[p]["v"]= v
        return loss