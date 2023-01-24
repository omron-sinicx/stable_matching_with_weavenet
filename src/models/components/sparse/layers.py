# sparse weavenet layers.
from torch_scatter import scatter_max #, scatter_min, scatter_mean
from torch_scatter.composite import scatter_softmax
from ..layers import ConcatMerger

class IndexSelectFormatter(nn.Module):
    def forward(self,
                x:torch.Tensor,
                vertex_id:torch.Tensor,
                dim:int = 0,
               )->torch.Tensor:
        return torch.index_select(x, dim, vertex_id)
        
class MaxPoolingAggregator(nn.Module):
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
        x_max, _ = scatter_max(x_sp.values(), vertex_id, dim)
        return self.formatter(x_max, vertex_id, dim)

class SetEncoderBase(nn.Module):
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
                vertex_id: torch.Tensor,
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

        
class SetEncoderPointNet(SetEncoderBase):
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
            MaxPoolingAggregator(),
            ConcatMerger(dim_feature=-1),
            second_process,
            **kwargs,
        )
        

class DualSoftmax(nn.Module):
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

class DualSoftmaxSqrt(DualSoftmax):
    r"""
    
    A variation of :obj:`DualSoftMax` for evenly weighting backward values for two streams.
    
    .. math::
        \text{DualSoftmaxSqrt}(x^{ab}_{ij}, x^{ba}_{ij}) = \sqrt{\text{DualSoftmax}(x^{ab}_{ij}, x^{ba}_{ij})}
    
    """
    epsilon:float=10**-7
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
        zab, zba = self.apply_softmax(xab, src_id, tar_id, xba=xba)
        return torch.clamp(zab*zba, self.epsilon).sqrt(), zab, zba

class DualSoftmaxFuzzyLogicAnd(DualSoftmax):
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

    