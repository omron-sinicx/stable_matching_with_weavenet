import torch
from .loss import loss_one2one_correlation_exp, loss_one2one_correlation, loss_one2many_penalty, loss_stability, loss_sexequality, loss_egalitarian, loss_balance
from .metric import is_one2one, is_stable, binarize, calc_all_fairness_metrics, count_blocking_pairs, PreferenceFormat

from typing import Optional, List

def _to_bcnm(m:torch.Tensor)->torch.Tensor:
    if m.dim()==3:
        return m.unsqueeze(1)
    assert(m.dim()==4)
    B, N, M, C = m.shape
    if C==1:
        return m.view(B, 1, N, M)
    return m.permute(0, 3, 1, 2).contiguous()

@torch.jit.ignore
def _range_jit_ignore(val:int, device:torch.device)->torch.Tensor:
    return torch.arange(start=0,end=val, dtype=torch.int64, device=device)
                       
def _select_channel(
        m: torch.Tensor, 
        loss_sm, 
        loss_fairness:Optional[torch.Tensor]=None)->List[torch.Tensor]:
        r"""
        Select a channel of matching as the best candidate to be validated/tested.


        Shape: 
            - m: :math:`(B, C, N, M)`
            - loss_one2one: :math:`(B, C)`
            - loss_stability: :math:`(B, C)`
            - loss_fairness: :math:`(B, C)`
            - output: :math:`(B, N, M)`

        Args:
            m: matching candidates
            loss_one2one: one2one loss.
            loss_stability: stability loss.
            loss_fairness: fairness-related loss.

        Returns:
            selected channel ids 
        """        

        # dummy
        B, C, N, M = m.shape
        
        if loss_fairness is None:
            # select minimum loss_sm
            pass
        else:
            # if loss_sm == 0, select minimum fairness loss. otherwise, select minimum loss_sm.
            pass
        
        indices = [_range_jit_ignore(B, m.device),
                   torch.zeros((B, ), dtype=torch.int64, device = m.device)]
        return indices
    
@torch.jit.ignore
def _access_by_indices(x:torch.Tensor, indices:List[torch.Tensor])->torch.Tensor:
    return x[indices]

@torch.jit.ignore
def _add_filtered_by_indices(x:torch.Tensor, indices:List[torch.Tensor], tar:torch.Tensor)->torch.Tensor:
    x[indices] += tar
    return x

class CriteriaStableMatching():
    def __init__(self, 
                 one2one_weight: float = 1.0,
                 stability_weight: float = 0.7,
                 fairness: str = 'sexequality',
                 fairness_weight: float = 0.1,
                 loss_one2one: str = 'correlation', # correlation | correlation_exp | maximize_sum
                 gate_fairness_loss: bool = False, # if True, consider fairness loss only when the condition was satisfied.
                ): 
        
        self.loss_one2one = loss_one2one
                    
        self.one2one_weight = one2one_weight
        self.stability_weight = stability_weight
        self.fairness_weight = fairness_weight
        
        self.fairness = fairness
        if fairness == 'sexequality':
            self.larger_is_better = False
        else: # 'egalitarian' or 'balance'
            self.larger_is_better = True
        self.fairness_weight = fairness_weight
        if self.fairness_weight == 0.0:
            self.fairness = None # self.fairness is None if fairness == None or fairness_weight == 0.0
            
        self.gate_fairness_loss = False
        if gate_fairness_loss:
            self.gate_fairness_loss = True
            
    def generate_criterion(self):      
        if self.loss_one2one == 'correlation':
            loss_one2one = loss_one2one_correlation
        elif self.loss_one2one == 'correlation_exp':
            loss_one2one = loss_one2one_correlation_exp
        elif self.loss_one2one == 'loss_one2many_penalty':
            loss_one2one = loss_one2many_penalty
        else:
            RuntimeError('Unknown loss_one2one function "{}".'.format(loss_one2one))
        
        def _criterion_sm(
            m: torch.Tensor, 
            sab: torch.Tensor, 
            sba_t: torch.Tensor,
            one2one_weight : float = self.one2one_weight,
            stability_weight: float = self.stability_weight,
        ):
            log = {}
            fut_o = torch.jit.fork(loss_one2one,m)#ab, mba_t)
            l_s = loss_stability(m, sab, sba_t)
            l_o = torch.jit.wait(fut_o)
            
            loss = one2one_weight * l_o
            log['loss_one2one'] = l_o
        
            loss += stability_weight * l_s
            log['loss_stability'] = l_s
                                            
            return loss, log
        
        if self.fairness is None:
            def criterion_no_fairness(
                m: torch.Tensor, 
                sab: torch.Tensor, 
                sba_t: torch.Tensor,
                one2one_weight : float = self.one2one_weight,
                stability_weight: float = self.stability_weight,
            ):
                m = _to_bcnm(m)
                loss, log = _criterion_sm(m, sab, sba_t, one2one_weight, stability_weight)
                if m.size(1)==1:
                    return loss, log, m[:,0]
                
                indices= _select_channel(m, loss)                
                return loss, log, _access_by_indices(m, indices)
            return criterion_no_fairness                
            
            
        if self.fairness == 'sexequality':            
            loss_fairness = loss_sexequality
        elif self.fairness == 'egalitarian':
            loss_fairness = loss_egalitarian
        elif self.fairness == 'balance':
            loss_fairness = loss_balance
        else:
            RuntimeError('Unknown fairness criterion "{}".'.format(self.fairness))

        def criterion(
            m: torch.Tensor,
            sab: torch.Tensor,
            sba_t: torch.Tensor,
            fairness_weight : float = self.fairness_weight,
            fairness_criterion_name : str = self.fairness_criterion_name,
            gate_fairness_loss : bool = self.gate_fairness_loss,
        ):
            m = _to_bcnm(m)
            loss, log = _criterion_sm(m, sab, sba_t)            
            
            l = loss_fairness(m, sab, sba_t) 
            
            if m.size(1)==1:
                loss += fairness_weight * l * ((loss.detach()<=0).max(torch.tensor([not gate_fairness_loss])).to(l.dtype))
                log[fairness_criterion_name] = l[:,0]
                return loss, log, m[:,0]
            
            indices = _select_channel(m, loss, l)
            m_selected = _access_by_indices(m, indices)
            # apply fairness loss only to the selected matching.
            l = _access_by_indices(l, indices)
            loss_selected = _access_by_indices(loss, indices)
            loss = _add_filtered_by_indices(loss, indices, fairness_weight * l * ((loss_selected.detach()<=0).max(torch.tensor([not gate_fairness_loss])).to(l.dtype)))            
            # equivalent to ...
            #  if not self.gate_fairness_loss:
            #    loss[indices]  = fairness_weight * l[indices]
            # else:
            #   loss[indices] = fairness_weight * l[indices] * (loss[indices].detach()<=0)
            log[fairness_criterion_name] = l
            return loss, log, m_selected
        
        return criterion
    
    @property
    def base_criterion_names(self):
        return ['loss_one2one', 'loss_stability']
    
    @property
    def fairness_criterion_name(self):
        return 'loss_{}'.format(self.fairness)        
    
    @staticmethod
    def metric(m: torch.Tensor, sab: torch.Tensor, sba_t: torch.Tensor):   
        mb = binarize(m)
        futs = [
            torch.jit.fork(is_one2one,mb),
            torch.jit.fork(is_stable, mb, sab, sba_t),
            torch.jit.fork(count_blocking_pairs, mb, sab, sba_t),
        ]
        log = {}        
        log['sexequality'], log['egalitarian'], log['balance']= calc_all_fairness_metrics(mb, sab, sba_t, pformat = PreferenceFormat.satisfaction)
        temp_one2one = torch.jit.wait(futs[0])
        temp_stable =torch.jit.wait(futs[1])
        log['is_one2one'] = temp_one2one
        log['is_stable'] = temp_stable
        log['is_success'] = temp_one2one * temp_stable
        log['num_blocking_pair'] = torch.jit.wait(futs[2])
        #log['sexequality'] = sexequality_cost(mb, sab, sba_t, pformat = PreferenceFormat.satisfaction)
        #log['egalitarian'] = egalitarian_score(mb, sab, sba_t, pformat = PreferenceFormat.satisfaction)
        #log['balance'] = balance_score(mb, sab, sba_t, pformat = PreferenceFormat.satisfaction)
        
        return log, mb
        
    @property
    def metric_names(self):
        return ['is_one2one', 'is_stable', 'is_success', 'num_blocking_pair', 'sexequality', 'egalitarian','balance']
    