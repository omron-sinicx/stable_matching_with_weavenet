import torch
from torch_match.loss import loss_one2one_correlation_exp, loss_one2one_correlation,loss_one2one_maximize_sum, loss_stability, loss_sexequality, loss_egalitarian, loss_balance
from torch_match.metric import is_one2one, is_stable, binarize, sexequality_cost, balance_score, egalitarian_score, count_blocking_pairs, PreferenceFormat


    
class CriteriaStableMatching():
    def __init__(self, 
                 one2one_weight: float = 1.0,
                 stability_weight: float = 0.7,
                 fairness: str = 'sexequality',
                 fairness_weight: float = 0.1,
                 loss_one2one: str = 'correlation_exp', # correlation | correlation_exp | maximize_sum
                ):    
        
        self.loss_one2one = loss_one2one
                    
        self.one2one_weight = one2one_weight
        self.stability_weight = stability_weight
        self.fairness_weight = fairness_weight
        
        self.fairness = fairness
        self.fairness_weight = fairness_weight
        if self.fairness_weight == 0.0:
            self.fairness = None # self.fairness is None if fairness == None or fairness_weight == 0.0
        
    def generate_criterion(self):      
        if self.loss_one2one == 'correlation':
            loss_one2one = loss_one2one_correlation
        elif self.loss_one2one == 'correlation_exp':
            loss_one2one = loss_one2one_correlation_exp
        elif  self.loss_one2one == 'maximize_sum':
            loss_one2one = loss_one2one_maximize_sum
        else:
            RuntimeError('Unknown loss_one2one function "{}".'.format(loss_one2one))
        
        def _criterion_sm(m: torch.Tensor, sab: torch.Tensor, sba: torch.Tensor,
                          one2one_weight : float = self.one2one_weight,
                          stability_weight: float = self.stability_weight,
                         ):
            log = {}
            l = loss_one2one(m)
            loss = one2one_weight * l
            log['loss_one2one'] = l
        
            l = loss_stability(m, sab, sba)
            loss += stability_weight * l
            log['loss_stability'] = l
        
            return loss, log
        
        if self.fairness is None:
            return _criterion_sm                
            
            
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
            sba: torch.Tensor,
            fairness_weight : float = self.fairness_weight,
            fairness_criterion_name : str = self.fairness_criterion_name,
        ):
        
            loss, log = _criterion_sm(m, sab, sba)
            
            l = loss_fairness(m, sab, sba)
            loss += fairness_weight * l
            log[fairness_criterion_name] = l
            return loss, log
        
        return criterion
    
    @property
    def base_criterion_names(self):
        return ['loss_one2one', 'loss_stability']
    
    @property
    def fairness_criterion_name(self):
        return 'loss_{}'.format(self.fairness)
    
    
    @staticmethod
    def metric(m: torch.Tensor, sab: torch.Tensor, sba: torch.Tensor):
        mb = binarize(m)
        temp_one2one = is_one2one(mb)
        temp_stable = is_stable(mb, sab, sba)
        log = {}
        log['is_one2one'] = temp_one2one
        log['is_stable'] = temp_stable
        log['is_success'] = temp_one2one * temp_stable
        log['num_blocking_pair'] = count_blocking_pairs(mb, sab, sba)
        log['sexequality'] = sexequality_cost(mb, sab, sba, pformat = PreferenceFormat.satisfaction)
        log['egalitarian'] = egalitarian_score(mb, sab, sba, pformat = PreferenceFormat.satisfaction)
        log['balance'] = balance_score(mb, sab, sba, pformat = PreferenceFormat.satisfaction)
        return log, mb
        
    @property
    def metric_names(self):
        return ['is_one2one', 'is_stable', 'is_success', 'num_blocking_pair', 'sexequality', 'egalitarian','balance']
    