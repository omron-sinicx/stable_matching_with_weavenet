from typing import Tuple, Optional
from typing_extensions import Literal

import torch
import torch.nn.functional as F
from .preference import PreferenceFormat, to_cost, batch_sum

__all__ = [
    'binarize',
    'is_one2one',
    'is_stable',
    'count_blocking_pairs',
    'sexequality_cost',
    'egalitarian_score',
    'balance_score',
    'calc_all_fairness_metrics',
    'MatchingAccuracy',
]

#@torch.jit.script
def binarize(m : torch.Tensor):
    r"""
    Binarizes each matrix in a batch.
        
    Shape:
        - m:  :math:`(B, N, M)`
        - output: :math:`(B, N, M)`
    Args:
        m: a continously-relaxed assignment between sides :math:`a` and :math:`b`, where |a|=N, |b|=M.       
    Returns:
        A binarized batched matrices.
    """
    na, nb = m.shape[-2:]
    if na >= nb:
        m = F.one_hot(m.argmax(dim=-1), num_classes=nb)
    else:
        m = F.one_hot(m.argmax(dim=-2), num_classes=na).t()
    return m

#@torch.jit.script
def is_one2one(m : torch.Tensor):
    r"""
    Checks whether each matrix in a batch m has no duplicated correspondence.
        
    Shape:
        - m:  :math:`(B, N, M)`
        - output: :math:`(B)`
    Args:
        m: a binary assignment between sides :math:`a` and :math:`b`, where |a|=N, |b|=M.       
    Returns:
        A binary bool vector.
    """
    return ~((torch.sum(m,dim=-2)>1).any(dim=-1) + (torch.sum(m,dim=-1)>1).any(dim=-1))

def _is_stable(m : torch.Tensor, sab : torch.Tensor, sba : torch.Tensor) -> bool:
    N = sab.shape[0]
    M = sba.shape[0]


    for c in range(M):
        sab_selected = sab[:,c:c+1] # to keep dimension, sab[:,c] is implemented as sab[:,c:c+1]
        sab_selected = sab_selected.repeat_interleave(M,dim=1)
        unsab = (m*torch.clamp(sab_selected-sab,min=0)).mean(dim=1)

        sba_selected = sba[c:c+1,:] # keep dimension.
        sba_selected = sba_selected.repeat_interleave(N,dim=0)
        _sba = sba_selected.t()
        _m = m[:,c:c+1]
        _m = _m.repeat_interleave(N,dim=1)
        unsba = (_m*torch.clamp(sba_selected-_sba,min=0)).mean(dim=0)
        envy = (unsab*unsba).sum()
        if envy>0:
            return False
    return True

#@torch.jit.script
def is_stable(m : torch.Tensor, sab : torch.Tensor, sba_t : torch.Tensor) -> torch.Tensor:
    r"""
    Checks whether each matrix in a batch m is a stable match or not.
        
    Shape:
        - m:  :math:`(B, N, M)`
        - sab: :math:`(B, N, M)`
        - sab: :math:`(B, M, N)`
        - output: :math:`(B)`
    Args:
        m: a binary (or continously-relaxed) assignment between sides :math:`a` and :math:`b`, where |a|=N, |b|=M.       
        sab: a satisfaction at matching of agents in side :math:`a` to side :math:`b`.  
        sba: a satisfaction at matching of agents in side :math:`b` to side :math:`a`.  
    Returns:
        A binary bool vector.
    """
    sba = sba_t.transpose(-1,-2)
    return torch.tensor([_is_stable(_m,_sab,_sba) for _m,_sab,_sba in zip(m, sab, sba)], dtype=torch.bool, device=m.device)

def _count_blocking_pairs(m : torch.Tensor, sab : torch.Tensor, sba : torch.Tensor):

    N = sab.shape[0]
    M = sba.shape[0]

    n_blocking_pair = 0

    for c in range(M):
        sab_selected = sab[:,c:c+1] # to keep dimension, sab[:,c] is implemented as sab[:,c:c+1]
        sab_selected = sab_selected.repeat_interleave(M,dim=1)
        #unsab = (m*torch.clamp(sab_selected-sab,min=0)).mean(dim=1)

        unsab_target = (m*(sab_selected-sab)>0).sum(dim=1) # the summuated value must be 0 or 1 due to multiplied m.
        #print("unsab_target: ", (m*(sab_selected-sab)>0))
        sba_selected = sba[c:c+1,:] # keep dimension.
        sba_selected = sba_selected.repeat_interleave(N,dim=0)
        _sba = sba_selected.t()
        _m = m[:,c:c+1]
        _m = _m.repeat_interleave(N,dim=1)
        #unsba = (_m*torch.clamp(sba_selected-_sba,min=0)).mean(dim=0)
        unsba_target = (_m*(sba_selected-_sba)>0).sum(dim=0) # 0 or 1 as unsab_target
        #print("unsba_target: ", (_m*(sba_selected-_sba)>0))
        n = (unsab_target * unsba_target).sum()
        #print("number of found blocking_pair: ",n)
        n_blocking_pair += n
        #envy = (unsab*unsba).sum()
        #print("envy: ",envy)
    return float(n_blocking_pair)

def count_blocking_pairs(m : torch.Tensor, sab : torch.Tensor, sba_t : torch.Tensor)->torch.Tensor:
    r"""
    Count the number of blocking pairs for each matrix in batch m.
        
    Shape:
        - m:  :math:`(B, N, M)`
        - sab: :math:`(B, N, M)`
        - sab: :math:`(B, M, N)`
        - output: :math:`(B)`
    Args:
        m: a binary (or continously-relaxed) assignment between sides :math:`a` and :math:`b`, where |a|=N, |b|=M.       
        sab: a satisfaction at matching of agents in side :math:`a` to side :math:`b`.  
        sba: a satisfaction at matching of agents in side :math:`b` to side :math:`a`.  
    Returns:
        A binary bool vector.
    """
    sba = sba_t.transpose(-1,-2)
    return torch.tensor([_count_blocking_pairs(_m,_sab,_sba) for _m,_sab,_sba in zip(m, sab, sba)], dtype=torch.float32, device=m.device)


def sexequality_cost(m : torch.Tensor, cab : torch.Tensor, cba_t : torch.Tensor, 
                     pformat : PreferenceFormat = PreferenceFormat.cost) -> torch.Tensor :
    if pformat != PreferenceFormat.cost:
        cab = to_cost(mat=cab, pformat=pformat, dim=-1)
        cba = to_cost(mat=cba, pformat=pformat, dim=-2)
    batch_size = m.size(0)
    return (batch_sum(m, cab, batch_size) - batch_sum(m, cba_t, batch_size)).abs()

def egalitarian_score(m : torch.Tensor, cab : torch.Tensor, cba_t : torch.Tensor, 
                     pformat: PreferenceFormat = PreferenceFormat.cost) -> torch.Tensor:
    if pformat != PreferenceFormat.cost:
        cab = to_cost(cab, pformat, dim=-1)
        cba_t = to_cost(cba_t, pformat, dim=-2)
    batch_size = m.size(0)
    return (batch_sum(m, cab, batch_size) + batch_sum(m, cba_t, batch_size)) # egalitarian cost = -1 * egalitarian score.

#@torch.jit.script
def balance_score(m : torch.Tensor, cab : torch.Tensor, cba_t : torch.Tensor, 
                     pformat: PreferenceFormat = PreferenceFormat.cost) -> torch.Tensor:
    if pformat != PreferenceFormat.cost:
        cab = to_cost(cab, pformat, dim=-1)
        cba_t = to_cost(cba_t, pformat, dim=-2)
    batch_size = m.size(0)
    return torch.stack([batch_sum(m, cab, batch_size), batch_sum(m, cba_t, batch_size)]).max(dim=0)[0]

def calc_all_fairness_metrics(m : torch.Tensor, cab : torch.Tensor, cba_t : torch.Tensor, 
                     pformat: PreferenceFormat = PreferenceFormat.cost) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pformat != PreferenceFormat.cost:
        cab = to_cost(cab, pformat, dim=-1)
        cba_t = to_cost(cba_t, pformat, dim=-2)
    batch_size = m.size(0)
    A = batch_sum(m, cab, batch_size)
    B = batch_sum(m, cba_t, batch_size)
    se = (A-B).abs()
    egal = A+B
    balance = (se+egal)/2
    #balance_ = torch.stack([batch_sum(m, cab, batch_size), batch_sum(m.transpose(-1,-2), cba, batch_size)]).max(dim=0)[0]
    #assert((balance ==  balance_).all())
    return se, egal, balance
