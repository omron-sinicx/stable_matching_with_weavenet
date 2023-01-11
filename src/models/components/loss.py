from typing import Tuple
import torch
from .preference import batch_sum

class MatchingCrossEntropy(torch.nn.CrossEntropyLoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor)->torch.Tensor:
        n, m = input.shape[-2:]
        input = input.view(-1, m) # (b, n, m) -> (b*n, m)
        target = target.view(-1,m) # (b, n, m) -> (b*n, m)
        target, input = self.erase_empty_target_rows(target, input)
        target_idx = target.argmax(dim=-1)
        return super().forward(input, target_idx)
    @staticmethod
    def erase_empty_target_rows(target: torch.Tensor, input:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        is_valid_row = target.sum(dim=-1)>0
        return target[is_valid_row], input[is_valid_row]
        
    
def loss_one2one_maximize_sum(m : torch.Tensor) -> torch.Tensor:
    '''
    this function assumes that $m$ is preliminary dual-softmaxed.
    '''
    n = min(m.size(-2), m.size(-1))
    return 1.0 - ((m.view(m.size(0), -1).sum(dim=-1) / n)).mean()

def loss_one2one_correlation_exp(m : torch.Tensor, epsilon:float=10**-7) -> torch.Tensor:
    m_exp = torch.clamp(m, epsilon).exp() # add epsilon to m for numerical stability.
    mc_norm = m_exp.norm(p=2,dim=-1,keepdim=True)
    mr_norm = m_exp.norm(p=2,dim=-2,keepdim=True)
    N,M = m.shape[-2:]
    Z = (N+M)/(2*N*M)
    
    batch_size = m.size(0)
    dM = ((m_exp)/mc_norm * (m_exp)/mr_norm).view(batch_size, -1).sum(dim=-1)*(Z)
    return 1 - dM.mean()

def loss_one2one_correlation(m : torch.Tensor, epsilon:float=10**-7) -> torch.Tensor:
    mc_norm = m.norm(p=2,dim=-1,keepdim=True)
    mr_norm = m.norm(p=2,dim=-2,keepdim=True)
    N,M = m.shape[-2:]
    Z = (N+M)/(2*N*M)
    
    batch_size = m.size(0)
    dM = ((m)/mc_norm * (m)/mr_norm).view(batch_size, -1).sum(dim=-1)*(Z)
    return 1 - dM.mean()


@torch.jit.script
def _loss_stability(m : torch.Tensor, sab : torch.Tensor, sba : torch.Tensor, epsilon:float=10**-7) -> torch.Tensor:
    '''
    originally proposed in [Shira Li, "Deep Learning for Two-Sided Matching Markets"](https://www.math.harvard.edu/media/Li-deep-learning-thesis.pdf)
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

def loss_stability(m : torch.Tensor, sab : torch.Tensor, sba : torch.Tensor) -> torch.Tensor:        
    return torch.stack([_loss_stability(_m,_sab,_sba) for _m, _sab, _sba in zip(m, sab, sba)]).mean()

def loss_sexequality(m : torch.Tensor, sab : torch.Tensor, sba : torch.Tensor) -> torch.Tensor:
    batch_size = m.size(0)
    # return |(m * sab).sum() - m.t() * sba).sum()|
    return (batch_sum(m, sab, batch_size) - batch_sum(m.transpose(-1,-2), sba, batch_size)).abs().mean()

def loss_egalitarian(m : torch.Tensor, sab : torch.Tensor, sba : torch.Tensor) -> torch.Tensor:
    batch_size = m.size(0) 
    return -  (batch_sum(m, sab, batch_size) + batch_sum(m.transpose(-1,-2), sba, batch_size)).mean()            

def loss_balance(m : torch.Tensor, sab : torch.Tensor, sba : torch.Tensor) -> torch.Tensor:
    batch_size = m.size(0)
    # return - average of min((m * sab).sum(), (m.t() * sba).sum())
    return - torch.stack([batch_sum(m, sab, batch_size), batch_sum(m.transpose(-1,-2), sba, batch_size)]).min(dim=0)[0].mean()

