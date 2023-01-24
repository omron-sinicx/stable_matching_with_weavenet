import numpy as np
import torch
from enum import Enum
import warnings

try:
    from functools import cached_property
    is_cp_supported = True
except:
    is_cp_supported = False

class PreferenceFormat(Enum):
    satisfaction = 1 # assume satisfaction is normalized in range(0.1, 1.0) by the numpy.linspace function.
    rank = 2
    cost = 3
    
def batch_sum(matching : torch.Tensor, mat : torch.Tensor, batch_size : int):
    return (mat * matching).view(batch_size, -1).sum(dim=-1)

    
def sat2rank(sat : torch.Tensor, dim = -1):
    return torch.argsort(sat, dim=dim, descending = True)

def cost2rank(cost : torch.Tensor, dim : int = -1):
    return torch.argsort(cost, dim=dim, descending = False)

def sat2cost(sat : torch.Tensor, dim : int = -1):
    n_opposites = sat.size(dim)
    return torch.round((n_opposites-1)*(1-(sat-0.1)/0.9)) + 1 # invert numpy.linspace

def cost2sat(cost : torch.Tensor, dim : int = -1, opposite_side_dim: int = -2):
    raise NotImplementedError("cost2sat is not implemented.")
    
def to_cost(mat : torch.Tensor, pformat : PreferenceFormat, dim : int = -1):
    if pformat == PreferenceFormat.cost:
        return mat
    if pformat == PreferenceFormat.satisfaction:
        return sat2cost(mat, dim)
    if pformat == PreferenceFormat.rank:
        raise RuntimeError("Impossible conversion: rank to cost")
    raise RuntimeError("Unsupported format")
    
def to_sat(mat : torch.Tensor, pformat : PreferenceFormat, dim : int = -1):
    if pformat == PreferenceFormat.cost:
        if dim == -1:
            opposite_side_dim = -2
        else:
            opposite_side_dim = dim + 1
        return cost2sat(mat, dim, opposite_side_dim)
    if pformat == PreferenceFormat.satisfaction:
        return mat
    if pformat == PreferenceFormat.rank:
        raise RuntimeError("Impossible conversion: rank to sat")
    raise RuntimeError("Unsupported format")

def to_rank(mat : torch.Tensor, pformat : PreferenceFormat, dim : int = -1):
    if pformat == PreferenceFormat.cost:
        return cost2rank(mat, dim=dim)
    if pformat == PreferenceFormat.satisfaction:
        return sat2rank(mat, dim=dim)
    if pformat == PreferenceFormat.rank:
        return mat
    raise RuntimeError("Unsupported format")
    
