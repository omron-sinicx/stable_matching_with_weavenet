# sparse weavenet layers.
import torch
from torch import nn
from typing import Optional, Callable, Tuple
from torch_scatter import scatter_max #, scatter_min, scatter_mean
from torch_scatter.composite import scatter_softmax
from ..layers import ConcatMerger

@torch.jit.ignore
def resampling_relaxed_Bernoulli(logits:torch.Tensor, tau:float)->torch.Tensor:
    sampler = torch.distributions.RelaxedBernoulli(tau, logits=logits)  
    return sampler.rsample()

def gumbel_sigmoid_logits(logits:torch.Tensor,
                          tau:float=1.,
                          hard:bool=False,
                  )->torch.Tensor:
    #sampler = MyRelaxedBernoulli(tau, logits=logits)   
    # y_soft = sampler.rsample()
    y_soft = resampling_relaxed_Bernoulli(logits, tau)
    if hard:
        # do resampling trick
        y_hard = (y_soft > 0.5).to(logits.dtype)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

def kthlargest_resampling(x: torch.Tensor, dim:int, tau:float, drop_rate:float)->torch.Tensor:
        y_soft = gumbel_sigmoid_logits(x, tau, hard=False)
        K = max(int(y_soft.size(dim) * (1.0-drop_rate)), 1)
        kth_val = y_soft.kthvalue(K, dim=dim, keepdim=True)[0]
        y_hard = (y_soft >= kth_val).to(x.dtype)
        return y_hard -y_soft.detach() + y_soft
                 
class LinearMaskInferenceOr(nn.Module):
    def __init__(self,
                 dim_src:int=-3,
                 dim_tar:int=-2,
                 drop_rate:float = 0.5,
                 tau:float = 1.0,
                )->None:
        r"""        
        Args:
            in_channels: the number of input channels.
            out_channels: the number of output channels at the second convolution.
            tau: the temperature of gumbel softmax
        """ 
        super().__init__()
        self.tau = tau
        self.dim_src = dim_src
        self.dim_tar = dim_tar
        self.drop_rate = drop_rate
        
    def build(self,
              input_channels:int,
              output_channels:int = 1,)->None:
        self.linear = nn.Linear(input_channels, output_channels, bias=True)
        
    
        
    def forward(self,
                xab: torch.Tensor,
                xba_t: torch.Tensor,
               )->torch.Tensor: 
        xab = self.linear.forward(xab)
        xab = kthlargest_resampling(xab, self.dim_src, self.tau, self.drop_rate)
        xba_t = self.linear.forward(xba_t)
        xba_t = kthlargest_resampling(xba_t, self.dim_tar, self.tau, self.drop_rate)
        y = xab + xba_t
        y[y==2.0] /= 2
        return y
    

class SparseDenseAdaptor():
    def __init__(self, mask:torch.Tensor):
        # (\ldots, N, M)
        
        self.shape = mask.shape[:-1]
        self.N, self.M = mask.shape[-3:-1]
        self.mask_lo = mask.view(-1, self.N, self.M)
        self.indices = torch.nonzero(self.mask_lo).t()
        self.src_vertex_id = self.indices[0]*self.N+self.indices[1]
        self.tar_vertex_id = self.indices[0]*self.M+self.indices[2]
        
    def _local_view(self,
                   x:torch.Tensor)->torch.Tensor:
        C = x.size(-1)
        return x.view(-1, self.N, self.M, C)
        
    def to_sparse(self,
                x: torch.Tensor)->torch.Tensor:
        # (\ldots, N, M, C)
        x = self._local_view(x)
        C = x.size(-1)
        values = x[self.mask_lo>0.5].view(-1, C)
        return values
    
    @torch.jit.ignore
    def to_dense(self,
                 x: torch.Tensor,
                 base: Optional[torch.Tensor]=None)->torch.Tensor:
        shape = self.shape + (x.size(-1),)
        x = torch.sparse_coo_tensor(self.indices, x, shape, device=x.device, dtype=x.dtype).to_dense()
        return x
        
    
class IndexSelectFormatter(nn.Module):
    def forward(self,
                x:torch.Tensor,
                vertex_id:torch.Tensor,
                dim:int = 0,
               )->torch.Tensor:
        return torch.index_select(x, dim, vertex_id)
        
class MaxPoolingAggregatorSp(nn.Module):
    def __init__(self):
        r"""
        Args:
            dim: the axis aggregated in the forward function.
        """
        super().__init__()
        self.formatter = IndexSelectFormatter()
        
    def forward(self, x_sp:torch.Tensor, 
                vertex_id:torch.Tensor, 
                dim:int = 0)->torch.Tensor:
        r"""
        Shape:
           - x: :math:`(\ldots, M, D)` if dim = -2, otherwise, the axis directed by dim should have M and aggregated while keeping dims.
           - output:  :math:`(\ldots, 1, D)`　 

        Args:
           x: an input tensor.

        Returns:
           x_aggregated

        """        
        x_max, _ = scatter_max(x_sp, vertex_id, dim)
        return self.formatter(x_max, vertex_id, dim)

class SetEncoderBaseSp(nn.Module):
    r"""
    
    Applies abstracted set-encoding process.
    
    .. math::
        \text{SetEncoderBase}(x) = \text{second_process}(\text{merger}(x, \text{aggregator}(\text{first_process}(x))))
    
    """
    def __init__(self, 
                 first_process: Callable[[torch.Tensor], torch.Tensor], 
                 aggregator: Callable[[torch.Tensor], torch.Tensor], 
                 merger: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 second_process: Callable[[torch.Tensor], torch.Tensor],
                 #return_vertex_feature:bool=False,
                ):
        r"""        
        Args:
           first_process: a callable (and typically trainable) object that converts a :math:`(B, D_{input}, N, M)` tensor  to :math:`(B, D_{mid}, N, M)`.
           aggregator: a callable object that aggregate :math:`M` edge features for each of :math:`N` vertices, results in a conversion of tensor from :math:`(B, D_{input}, N, M)` to :math:`(B, D_{mid}, N, 1)`.
           merger: a callable object that merge  :math:`(B, D_{input}, N, M)` edge features and  :math:`(B, D_{mid}, N, 1)` vertex features into  :math:`(B, D_{merged}, N, M)`.
           second_process: a callable (and typically trainable) object that converts a :math:`(B, D_{merged}, N, M)` tensor  to :math:`(B, D_{output}, N, M)`.

        """        
        super().__init__()
        self.first_process = first_process
        self.aggregator = aggregator
        self.merger = merger
        self.second_process = second_process
        #self.return_vertex_feature = return_vertex_feature
        
    def forward(self, 
                x:torch.Tensor,
                vertex_id: torch.Tensor, # the only difference from dense SetEncoderBase
               )->torch.Tensor:
        r"""
        Shape:
           - x: :math:`(\ldots)` (not defined with this abstractive class)
           - output:  :math:`(\ldots)`　 (not defined with this abstractive class)

        Args:
           x: an input tensor.

        Returns:
           z_edge_features, z_vertex_features

        """        
        z = self.first_process(x)
        z_vertex = self.aggregator(z, vertex_id, dim=0)
        z = self.merger(x, z_vertex)
        return self.second_process(z)

        
class SetEncoderPointNetSp(SetEncoderBaseSp):
    def __init__(self, in_channels:int, mid_channels:int, output_channels:int, **kwargs):
        r"""        
        Args:
            in_channels: the number of input channels.
            mid_channels: the number of output channels at the first convolution.
            out_channels: the number of output channels at the second convolution.
           
        """ 
        first_process = nn.Linear(in_channels, mid_channels)
        second_process = nn.Linear(in_channels + mid_channels, output_channels, bias=False)    
            
        super().__init__(
            first_process, 
            MaxPoolingAggregatorSp(),
            ConcatMerger(dim_feature=-1),
            second_process,
            **kwargs,
        )
        
StreamAggregatorSp = Callable[
    [torch.Tensor,torch.Tensor,torch.Tensor, Optional[torch.Tensor]],
    Tuple[torch.Tensor,torch.Tensor,torch.Tensor]]
class DualSoftmaxSp(nn.Module):
    r"""
    
    Applies the dual-softmax calculation to a batched matrices. DualSoftMax is originally proposed in `LoFTR (CVPR2021) <https://zju3dv.github.io/loftr/>`_. 
    
    .. math::
        \text{DualSoftmax}(x^{ab}_{ij}, x^{ba}_{ij}) = \frac{\exp(x^{ab}_{ij})}{\sum_j \exp(x^{ab}_{ij})} * \frac{\exp(x^{ba}_{ij})}{\sum_i \exp(x^{ba}_{ij})} 

    In original definition, always :math:`x^{ba}=x^{ab}`. This is an extensional implementation that accepts :math:`x^{ba}\neq x^{ab}` to input the two stream outputs of `WeaveNet`. 
    
    """        
    def apply_softmax(self,
                      xab:torch.Tensor, 
                      src_id:torch.Tensor,
                      tar_id:torch.Tensor,
                      xba:Optional[torch.Tensor]=None,
                     )->Tuple[torch.Tensor, torch.Tensor]:
        if xba is None:
            xba = xab
        zab = scatter_softmax(xab, src_id, dim=0)
        zba = scatter_softmax(xba, tar_id, dim=0)
        return zab, zba
    
    

    def forward(self, 
                xab:torch.Tensor, 
                src_id:torch.Tensor,
                tar_id:torch.Tensor,
                xba:Optional[torch.Tensor] = None,
               )->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        r""" Calculate the dual softmax for batched matrices.
                
        Shape:
           - xab: :math:`(\ldots * N_1 * M_1, C)`
           - src_id: :math:`(\ldots * N_1 * M_1)`
           - tar_id: :math: `(\ldots * N_1 * M_1)`
           - xba: :math:`(\ldots * N_1 * M_1, C)`
           - output:  :math:`(\ldots * N_1 * M_1, C)`
           
        Args:
           xab: 1st batched matrices.
           
           xba: 2nd batched matrices. x_ab is used as (not-transposed) x_ba if None. This option corresponds to the original implementation of LoFTR.
           
           is_xba_transposed: set False if :math:`(N_1, M_1)==(N_2, M_2)` and set True if :math:`(N_1, M_1)==(M_2, N_2). Default: False
           
        Returns:
           values mab * mba_t, mab (=softmax(xab, dim=-2)), mba_t (=softmax(xba_t, dim=-1)
           
        """
        zab, zba = self.apply_softmax(xab, src_id, tar_id, xba=xba)
        return zab * zba, zab, zba

class DualSoftmaxSqrtSp(DualSoftmaxSp):
    r"""
    
    A variation of :obj:`DualSoftMax` for evenly weighting backward values for two streams.
    
    .. math::
        \text{DualSoftmaxSqrt}(x^{ab}_{ij}, x^{ba}_{ij}) = \sqrt{\text{DualSoftmax}(x^{ab}_{ij}, x^{ba}_{ij})}
    
    """
    def forward(self, 
                xab:torch.Tensor, 
                src_id:torch.Tensor,
                tar_id:torch.Tensor,
                xba:Optional[torch.Tensor] = None,
               )->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        r""" Calculate the dual softmax for batched matrices.
        Shape:
           - xab: :math:`(\ldots * N_1 * M_1, C)`
           - src_id: :math:`(\ldots * N_1 * M_1)`
           - tar_id: :math: `(\ldots * N_1 * M_1)`
           - xba: :math:`(\ldots * N_1 * M_1, C)`
           - output:  :math:`(\ldots * N_1 * M_1, C)`
           
        Args:
            x_ab: 1st batched matrices
            
            x_ba: 2nd batched matrices. x_ab is used as (not-transposed) x_ba if None. This option corresponds to the original implementation of LoFTR.
            
            is_ba_transposed: True if rows1==cols2 and cols1==rows2. False if rows1 == rows2 and cols1 == cols2. Default: True
       Returns:
           values (mab * mba_t).sqrt(), mab (=softmax(xab, dim=-2)), mba_t (=softmax(xba_t, dim=-1)
        """
        epsilon:float=10**-7
        zab, zba = self.apply_softmax(xab, src_id, tar_id, xba=xba)
        return torch.clamp(zab*zba, epsilon).sqrt(), zab, zba

class DualSoftmaxFuzzyLogicAndSp(DualSoftmaxSp):
    r"""
    
    Applies the calculation proposed in `Shira Li, "Deep Learning for Two-Sided Matching Markets" <https://www.math.harvard.edu/media/Li-deep-learning-thesis.pdf>`_.
    
    .. math::
        \text{DualSoftmaxFuzzyLogicAnd}(x^{ab}_{ij}, x^{ba}_{ij}) = \min(\frac{\exp(x^{ab}_{ij})}{\sum_j \exp(x^{ab}_{ij})},  \frac{\exp(x^{ba}_{ij})}{\sum_i \exp(x^{ba}_{ij})})

    
    """
    def forward(self,
                xab:torch.Tensor, 
                xba:Optional[torch.Tensor]=None, 
                is_xba_transposed:bool=True)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        r""" Calculate the dual softmax for batched matrices.
                
        Shape:
           - xab: :math:`(\ldots * N_1 * M_1, C)`
           - src_id: :math:`(\ldots * N_1 * M_1)`
           - tar_id: :math: `(\ldots * N_1 * M_1)`
           - xba: :math:`(\ldots * N_1 * M_1, C)`
           - output:  :math:`(\ldots * N_1 * M_1, C)`
           
        Args:
           x_ab: 1st batched matrices.
           
           x_ba: 2nd batched matrices. x_ab is used as (not-transposed) x_ba if None. This option corresponds to the original implementation of LoFTR.
           
           is_ba_transposed: set False if :math:`(N_1, M_1)==(N_2, M_2)` and set True if :math:`(N_1, M_1)==(M_2, N_2). Default: False
           
        Returns:
           values torch.min(mab, mba_t), mab (=softmax(xab, dim=-2)), mba_t (=softmax(xba_t, dim=-1)
           
        """
        zab, zba = self.apply_softmax(xab, src_id, tar_id, xba=xba)
        return zab.min(zba), zab, zba            

if __name__ == "__main__":
    #_ = WeaveNetOldImplementation(2, 2,1)
    _ = WeaveNet(
            WeaveNetHead6(1,), 2, #input_channel:int,
                 [4,8,16], #out_channels:List[int],
                 [2,4,8], #mid_channels:List[int],1,2,2)
                 calc_residual=[False, False, True],
                 keep_first_var_after = 0,
                 stream_aggregator = DualSoftMaxSqrt())
    