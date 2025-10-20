import torch
import torch.nn as nn 
import os,typing
def save_checkpoint(model:torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    iteration: int, 
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    model_checkpoint = model.state_dict()
    optim_checkpoint = optimizer.state_dict()
    dict = {
        "it": iteration,
        "model": model_checkpoint,
        "optim": optim_checkpoint,
    }
    torch.save(dict,out)

def load_checkpoint(src, model:torch.nn.Module, optimizer:torch.optim.Optimizer) :
    dict = torch.load(src) 
    iter = dict.get("it")
    model.load_state_dict( dict.get("model"))
    optimizer.load_state_dict(dict.get("optim"))
    return iter