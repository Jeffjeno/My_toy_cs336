import numpy as np
import torch
from typing import List
def data_loading(x: List[int] , batch_size :int , context_length : int , device : torch.device):
    # i + context_length + 1 <= n
    ix = np.random.randint(0, len(x) - context_length ,size = batch_size)
    inp = [ x[i : i + context_length ] for i in ix ]
    tar = [ x[ i+1 : i+ context_length + 1] for i in ix] 
    inp = np.array(inp)
    tar = np.array(tar)
    inp_tensor = torch.tensor(inp,dtype=torch.long , device= device)
    tar_tensor = torch.tensor(tar,dtype=torch.long , device= device)
    return (inp_tensor,tar_tensor)