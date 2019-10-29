import torch
import numpy as np 

def discounted_cumsum_torch(x, discount):
    ret = torch.zeros_like(x)
    ret[-1] = x[-1]
    for t in range(len(x) - 2, -1, -1):
        ret[t] = x[t] + discount * ret[t + 1]
    return ret