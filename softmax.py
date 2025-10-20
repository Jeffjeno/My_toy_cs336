import torch

def softmax(x : torch.Tensor , d : int):
    x_pre = x - x.max(dim=d,keepdim=True).values
    x_pre = torch.exp(x_pre)
    x_sum = x_pre.sum(dim = d ,keepdim= True)
    return x_pre/x_sum