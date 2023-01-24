from typing import Tuple, Optional
import torch
from .preference import batch_sum
        
__all__ = [
    'loss_one2one_correlation',
    'loss_one2many_penalty',
    'loss_stability',
    'loss_sexequality',
    'loss_egalitarian',
    'loss_balance',
]

def set_mba_t(func):
    def wrapper(mab: torch.Tensor, mba_t:Optional[torch.Tensor]=None)->torch.Tensor:
        if mba_t is None:
            mba_t = mab        
        return func(mab, mba_t)
    return wrapper

@set_mba_t
def loss_one2many_penalty(mab : torch.Tensor, mba_t:torch.Tensor) -> torch.Tensor:    
    return (torch.clamp(mab.sum(dim=-2)-1,0) + torch.clamp(mba_t.sum(dim=-1)-1,0)).mean(dim=-1)

'''
def loss_one2one_maximize_sum(m : torch.Tensor) -> torch.Tensor:
    #this function assumes that $m$ is preliminary dual-softmaxed.
    n = min(m.size(-2), m.size(-1))
    return 1.0 - ((m.view(m.size(0), -1).sum(dim=-1) / n))
'''

@set_mba_t
def loss_one2one_correlation_exp(mab : torch.Tensor, mba_t:torch.Tensor, epsilon:float=10**-7) -> torch.Tensor:
    mab_exp = torch.clamp(mab, epsilon).exp() # add epsilon to m for numerical stability.
    mba_t_exp = torch.clamp(mba_t, epsilon).exp() # add epsilon to m for numerical stability.
    mc_norm = mab_exp.norm(p=2,dim=-2,keepdim=True)
    mr_norm = mba_t_exp.norm(p=2,dim=-1,keepdim=True)
    N,M = mab.shape[-2:]
    Z = (N+M)/(2*N*M)
    
    batch_size = m.size(0)
    dM = ((mab_exp)/mc_norm * (mba_t_exp)/mr_norm).view(batch_size, -1).sum(dim=-1).mean(dim=-1)*(Z)
    return 1.0 - dM

@set_mba_t
def loss_one2one_correlation(mab : torch.Tensor, mba_t:torch.Tensor, epsilon:float=10**-7) -> torch.Tensor:
    r"""
    Calculates a loss to maintain :math:`m` to be a doubly-stochastic matrix, which is a contiously-relaxed one-to-one matching.
    
    .. math::
        \mathcal{L}_{\rm one2one}(m) = 1 - \frac{N+M}{2NM} \sum_{(i,j)\in N\times M} (\frac{m_{i,j}}{||m_{*,j}||_2} * \frac{m_{i,j}}{||m_{i,*}||_2})
    
    Shape:
        - m:  :math:`(B, N, M)`
    Args:
        m: a continously-relaxed assignment between sides :math:`a` and :math:`b`, where |a|=N, |b|=M.       
    Returns:
        The calculated :math:`\mathcal{L}_{\rm one2one}(m)`.
        
   　．． note::
        The coefficient :math:`\frac{N+M}{2NM}` can be :math:`\frac{1}{\max(N,M)}` for :math:`N\neq M` cases?
        
    """    
    mc_norm = torch.clamp(mab.norm(p=2,dim=-2,keepdim=True), epsilon)
    mr_norm = torch.clamp(mba_t.norm(p=2,dim=-1,keepdim=True), epsilon)
    N,M = mab.shape[-2:]
    Z = (N+M)/(2*N*M)
    
    batch_size = mab.size(0)
    dM = ((mab)/mc_norm * (mba_t)/mr_norm).view(batch_size, -1).sum(dim=-1)*(Z)
    return 1.0 - dM

def _loss_stability(m : torch.Tensor, sab : torch.Tensor, sba : torch.Tensor, epsilon:float=10**-7) -> torch.Tensor:
    '''
    originally proposed in `Shira Li, "Deep Learning for Two-Sided Matching Markets" <https://www.math.harvard.edu/media/Li-deep-learning-thesis.pdf>`_
    '''
    uns=m.new_zeros(1)
    if not (sab.shape[0]==sba.shape[1] and sab.shape[0]==m.shape[0]):
        print(sab.shape, sba.shape, m.shape)
    assert sab.shape[0]==sba.shape[1] and sab.shape[0]==m.shape[0]
    N = m.shape[0]
    M = m.shape[1]
    for c in range(M):
        sab_selected = sab[:,c:c+1] # to keep dimension, sab[:,c] is implemented as sab[:,c:c+1]
        sab_selected = torch.repeat_interleave(sab_selected,M,dim=1)
        # c           i=0         i=1         i=2
        # 0 tensor([1.0000e-07, 1.0000e-07, 1.0000e-07])
        unsab = (m*torch.clamp(sab_selected-sab,epsilon)).sum(dim=1)

        sba_selected = sba[c:c+1,:] # keep dimension.
        sba_selected = torch.repeat_interleave(sba_selected,N,dim=0)
        _sba = sba_selected.t()

        _m = m.new_zeros(N,N)
        _m += m[:,c:c+1]


        # c           i=0         i=1         i=2
        # 0 tensor([1.0000e-07, 1.0000e-07, 1.0000e-07])
        unsba = (_m*torch.clamp(sba_selected-_sba,epsilon)).sum(dim=0)
        # Admarl unsab*unsba unsab[0]*unsba[0], unsab[1]*unsba[1],unsab[2]*unsba[2],
        uns += (unsab*unsba).sum()
    return uns

def loss_stability(m : torch.Tensor, sab : torch.Tensor, sba_t : torch.Tensor) -> torch.Tensor:
    r"""
    Calculates a loss to minimize violation of stability constraints of stable marriage problem, originally proposed in 
    `Shira Li, "Deep Learning for Two-Sided Matching Markets" <https://www.math.harvard.edu/media/Li-deep-learning-thesis.pdf>`_
    as `expected ex post stability violation`.
    
    .. math::
        \text{envy}^{ab}_j(m, s^{ab}) &=& \sum_{n\in N\backslash\{i\}}m_{i,j}\max(s^{ab}_{i,j}-s^{ab}_{n,j},0) \\
        \text{envy}^{ba}_i(m,, s^{ba}) &=& \sum_{m\in M\backslash\{j\}}m^\top_{j,i}\max(s^{ba}_{j,i}-s^{ba}_{m,i},0) \\
        \mathcal{L}_{\rm stability}(m, s^{ab}, s^{ba}) &=& \sum_{(i,j)\in N\times M} \text{envy}^{ab}_j(m, s^{ab}) * \text{envy}^{ba}_j(m, s^{ba})
    
    Shape:
        - m:  :math:`(B, N, M)`
        - sab: :math:`(B, N, M)`
        - sba_t: :math:`(B, N, M)`
    Args:
        m: a continously-relaxed assignment between sides :math:`a` and :math:`b`, where |a|=N, |b|=M.       
        sab: a satisfaction at matching of agents in side :math:`a` to side :math:`b`.  
        sba_t: a satisfaction at matching of agents in side :math:`b` to side :math:`a` (transposed).  
    Returns:
        The calculated `expected ex post stability violation` of :math:`\mathcal{L}_{\rm stability}(m, s^{ab}, s^{ba})`.
        
    """
    sba = sba_t.transpose(-1,-2)
    return torch.stack([_loss_stability(_m,_sab,_sba) for _m, _sab, _sba in zip(m, sab, sba)]).squeeze(-1)

def loss_sexequality(m : torch.Tensor, sab : torch.Tensor, sba_t : torch.Tensor) -> torch.Tensor:
    r"""
    Calculates a loss to minimize `sex-equality cost <https://core.ac.uk/download/pdf/160454594.pdf>`_.
    
    .. math::
       S^{ab}(m) &=& \sum_{(i,j)\in N\times M} m_{i,j}s^{ab}_{i,j} \\
       S^{ba}(m) &=& \sum_{(i,j)\in N\times M} m^\top_{j,i}s^{ba}_{j,i} \\
        \mathcal{L}_{\rm sexequality}(m, s^{ab}, s^{ba}) &=& 
        | S^{ab}(m) - S^{ba}(m)|
    
    Shape:
        - m:  :math:`(B, N, M)`
        - sab: :math:`(B, N, M)`
        - sba_t: :math:`(B, N, M)`
    Args:
        m: a continously-relaxed assignment between sides :math:`a` and :math:`b`, where |a|=N, |b|=M.       
        sab: a satisfaction at matching of agents in side :math:`a` to side :math:`b`.  
        sba_t: a satisfaction at matching of agents in side :math:`b` to side :math:`a` (transposed).  
    Returns:
        Batch-wise mean of :math:`\mathcal{L}_{\rm sexequality}(m, s^{ab}, s^{ba})`.
    """
    batch_size = m.size(0)
    return (batch_sum(m, sab, batch_size) - batch_sum(m, sba_t, batch_size)).abs()

def loss_egalitarian(m : torch.Tensor, sab : torch.Tensor, sba_t : torch.Tensor) -> torch.Tensor:
    r"""
    Calculates a loss to minimize `egalitarian cost <https://core.ac.uk/download/pdf/160454594.pdf>`_.
    
    .. math::
        \mathcal{L}_{\rm egalitarian}(m, s^{ab}, s^{ba}) = S^{ab}(m) + S^{ba}(m),
    
    where :math:`S^{ab}(m)` and  :math:`S^{ba}(m)` are defined with :obj:`loss_sexequality`.
    
    Shape:
        - m:  :math:`(B, N, M)`
        - sab: :math:`(B, N, M)`
        - sba_t: :math:`(B, N, M)`
        - output: :math: `(B)`
    Args:
        m: a continously-relaxed assignment between sides :math:`a` and :math:`b`, where |a|=N, |b|=M.       
        sab: a satisfaction at matching of agents in side :math:`a` to side :math:`b`.  
        sba_t: a satisfaction at matching of agents in side :math:`b` to side :math:`a` (transposed).  
    Returns:
        loss :math:`\mathcal{L}_{\rm egalitarian}(m, s^{ab}, s^{ba})`.
    """    
    batch_size = m.size(0) 
    return -  (batch_sum(m, sab, batch_size) + batch_sum(m, sba_t, batch_size))

def loss_balance(m : torch.Tensor, sab : torch.Tensor, sba_t : torch.Tensor) -> torch.Tensor:
    r"""
    Calculates a loss to minimize `balance cost <https://papers.nips.cc/paper/2019/hash/cb70ab375662576bd1ac5aaf16b3fca4-Abstract.html>`_.
    
    .. math::
        \mathcal{L}_{\rm balance}(m, s^{ab}, s^{ba}) = min(S^{ab}(m), S^{ba}(m)),
    
    where :math:`S^{ab}(m)` and  :math:`S^{ba}(m)` are defined with :obj:`loss_sexequality`.
    
    .. note::
        It is also known that
        
        :math:`\mathcal{L}_{\rm balance}(m, s^{ab}, s^{ba}) = \frac{\mathcal{L}_{\rm sexequality}(m, s^{ab}, s^{ba})+\mathcal{L}_{\rm egalitarian}(m, s^{ab}, s^{ba})}{2}`.
        
    
    Shape:
        - m:  :math:`(B, N, M)`
        - sab: :math:`(B, N, M)`
        - sba_t: :math:`(B, N, M)`
        - output: :math: `(B)`

    Args:
        m: a continously-relaxed assignment between sides :math:`a` and :math:`b`, where |a|=N, |b|=M.       
        sab: a satisfaction at matching of agents in side :math:`a` to side :math:`b`.  
        sba: a satisfaction at matching of agents in side :math:`b` to side :math:`a`.  
    Returns:
        loss :math:`\mathcal{L}_{\rm balance}(m, s^{ab}, s^{ba})`.
    """    
    batch_size = m.size(0)
    # return - average of min((m * sab).sum(), (m.t() * sba).sum())
    return - batch_sum(m, sab, batch_size).min(batch_sum(m, sba_t, batch_size))

