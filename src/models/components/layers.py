import torch
import torch.nn as nn

from typing import Optional, Tuple, List, Callable


from torch.nn.modules.batchnorm import _BatchNorm

class BatchNormXXC(nn.Module):
    r"""
    
    Applies :class:`BatchNorm1d` to :math:`(\ldots, C)`-shaped tensors. This module is prepered since :class:`nn.BatchNorm2d` assumes the input format of :math:`(B, C, H, W)` but if kernel size is 1, :class:`nn.Conv2d` to :math:`(B, C, H, W)` is slower than Linear to :math:`(B, H, W, C)`, which is our case for bipartite-graph edge embedding of :math:`(B, N, M, C)`.
    
    
    **Example of Usage**::
    
        # assume batch_size=8, the problem instance size is 5x5,  and each edge feature is 32 channels.
        B, N, M, C = 8, 5, 5, 32        
        linear = nn.Linear(32, 64)
        bn = BatchNormXXC(64)
        x = torch.rand((B, N, M, C), dtype=torch.float) # prepare a random input.
        x = linear(x)
        x = bn(x)
    
    """
    
    def __init__(self, C)->None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=C)
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        r"""
        Shape:
           - x: :math:`(\ldots, C)`
           - output:  :math:`(\ldots, C)`

        Args:
           x: target variable.

        Returns:
           x: batch-normed variable.

        """
        shape = x.shape
        x = self.bn(x.view(-1,shape[-1]))
        return x.view(shape)
        
class Interactor(nn.Module):
    r"""
    
    Abstract :class:`CrossConcat` and any other interactor between feature blocks of two stream architecture. It must have a function :func:`output_channels` to report its resultant feature's output channels (estimated based on the :class:`input_channels`).    
    
    """    
    def output_channels(self, input_channels:int)->torch.Tensor:        
        return input_channels
    def forward(self, 
                xab: torch.Tensor, 
                xba_t:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        # do nothing
        return xab, xba_t
    
class CrossConcat(Interactor):
    r"""
    
    Applies cross-concatenation introduced in `Edge-Selective Feature Weaving for Point Cloud Matching <https://arxiv.org/abs/2202.02149>`_
    
    .. math::
        \text{CrossConcat}([x^{ab}, {x^{ba}}^\top]) = [\text{cat}([x^{ab}, {x^{ba}}^\top], dim=-1), \text{cat}([{x^{ba}}^\top,x^{ab}], dim=-1)]
        
    
    **Example of Usage**::
    
        # prepare an instance of this class.
        interactor = CrossConcat()
        
        # assume batch_size=8, the problem instance size is 6x5,  and each edge feature is 32 channels.
        B, N, M, C = 8, 6, 5, 32        
        
        xab = torch.rand((B, N, M, C), dtype=torch.float) # prepare a random input. # NxM
        xba = torch.rand((B, M, N, C), dtype=torch.float) # prepare a random input. # MxN
        xba_t = xba.transpose(1,2)
        zab, zba_t = interactor(xab, xba_t)
        assert(xab.size(-1)*2 == 2*C)
        assert(xba_t.size(-1)*2 == 2*C)
        assert(interactor.output_channels(C)==2*C)

    """
    def __init__(self, dim_feature:int=-1):
        super().__init__()
        self.dim_feature = dim_feature
    
    def output_channels(self, input_channels:int)->torch.Tensor:
        r"""
        Args:
           input_channels: assumed input channels.
           
        Returns:
           output_channels calculated based on the args.

        """        
        return input_channels*2
        
    def forward(self, 
                xab: torch.Tensor, 
                xba_t:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Shape:
           - xab: :math:`(\ldots, C)`
           - xba_t: :math:`(\ldots, C)`
           - output(zab, zba_t):  :math:`[(\ldots, 2*C),(\ldots, 2*C)]`

        Args:
           xab: batched feature map, typically with the size of (B, N, M, C) where ij-th feature at :math:`(i, j)\in N \times M` represent edges from side `a` to `b`.
           
           xba_t: batched feature map with the same shape with xab, and represent edges from side `b` to `a`.

        Returns:
           (zab, zba_t) calculated as :math:`\text{CrossConcat}([x^{ab}, {x^{ba}}^\top])` .

        """
        zab_fut = torch.jit.fork(torch.cat, [xab,xba_t],dim=self.dim_feature)
        zba_t = torch.cat([xba_t,xab],dim=self.dim_feature)
        zab = torch.jit.wait(zab_fut)
        return (zab, zba_t)


class CrossDifferenceConcat(Interactor):
    r"""
    
    Applies cross-concatenation of mean and difference (experimental). 
    
    .. math::
        \text{CrossConcat}([x^{ab}, {x^{ba}}^\top]) = [\text{cat}([x^{ab}, {x^{ba}}^\top], dim=-1), \text{cat}([{x^{ba}}^\top,x^{ab}], dim=-1)]
        
    .. note::
        This class was not very effective with stable matching test.
    """
    def __init__(self, dim_feature:int=-1):
        super().__init__()
        self.dim_feature = dim_feature
    def output_channels(self, input_channels:int)->torch.Tensor:
        return input_channels * 2
        
    def forward(self, 
                xab: torch.Tensor, 
                xba_t:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Shape:
           - xab: :math:`(\ldots, C)`
           - xba_t: :math:`(\ldots, C)`
           - output(zab, zba_t):  :math:`[(\ldots, 2*C),(\ldots, 2*C)]`

        Args:
           xab: batched feature map, typically with the size of (B, N, M, C) where ij-th feature at :math:`(i, j)\in N \times M` represent edges from side `a` to `b`.
           
           xba_t: batched feature map with the same shape with xab, and represent edges from side `b` to `a`.

        Returns:
           (zab, zba_t) calculated as :math:`\text{CrossConcat}([x^{ab}, {x^{ba}}^\top])` .

        """
        #merged = xab.min(xba_t)
        zab_fut = torch.jit.fork(torch.cat, [xab, xab - xba_t],dim=self.dim_feature)
        zba_t = torch.cat([xba_t, xba_t - xab],dim=self.dim_feature)
        zab = torch.jit.wait(zab_fut)
        return (zab, zba_t)
    

class RepeatFormatter(nn.Module):
        
    def forward(self, 
                x_vertex:torch.Tensor,
                x_edge_shape:List[int],
                dim_target:int)->torch.Tensor:            
        r"""
        Reformat a feature to a specific shape by repeat the value in `dim_target` dimension.

        Shape:
           - x_vertex: :math:`(\ldots, 1, \ldots)`
           - x_edge_shape: :math:`[\ldots, M, \ldots]` where :math:`M` is the number of elements at `dim_target`
           - dim_target: the repeating target dimension.
           - output: :math:`(\ldots, M, \ldots)`
           
        Args:
           x_vertex: a tensor of vertex-wise features
           x_edge_shape: a shape of edge-wise feature tensor

        Returns:
           x_reshaped
        """
        ndim = len(x_edge_shape)
        if ndim - x_vertex.dim()==1:
            x_vertex = x_vertex.unsqueeze(dim_target)
        else:
            assert(ndim == x_vertex.dim())
            #assert(x_vertex.shape[dim_target]==1)

        tar_shape = list(x_vertex.shape)
        tar_shape[dim_target] = x_edge_shape[dim_target]
        return x_vertex.expand(tar_shape)

class MaxPoolingAggregator(nn.Module):
    def __init__(self, dim:int=-2):
        r"""
        Args:
            dim: the axis aggregated in the forward function.
        """
        super().__init__()
        self.formatter = RepeatFormatter()
        
    def forward(self, x:torch.Tensor, dim_target:int)->torch.Tensor:
        r"""
        Shape:
           - x: :math:`(\ldots, M, D)` if dim = -2, otherwise, the axis directed by dim should have M and aggregated while keeping dims.
           - output:  :math:`(\ldots, 1, D)`　 

        Args:
           x: an input tensor.

        Returns:
           x_aggregated

        """        
        return self.formatter(x.max(dim=dim_target, keepdim=True)[0], x.shape, dim_target)


        
class ConcatMerger(nn.Module):   
    def __init__(self, dim_feature:int=-1):
        super().__init__()
        self.dim_feature = dim_feature
        
    def forward(self,
               x_edge:torch.Tensor, x_vertex:torch.Tensor)->torch.Tensor:
        r"""
        Applies concatenation to edge feature and vertex feature

        .. math::
            \text{Concat}(x_{edge}, x_{vertex}) = \text{cat}([x_{edge}, x_{vertex}]),
        Shape:
           - x_edge: :math:`(\ldots, D_{edge}, N, M)`
           - x_vertex: :math:`(\ldots, D_{vertex}, N, M)`
           - output:  :math:`(\ldots, D_{edge}+D_{vertex}, N, M)`　 
        Args:
           x_edge: a batched tensor of edge-wise features
           x_vertex: a batched tensor of vertex-wise features

        Returns:
           x_merged
        """
        return torch.cat([x_edge, x_vertex], dim=self.dim_feature)
    
class DifferenceConcatMerger(ConcatMerger):    
    def forward(self,
               x_edge:torch.Tensor, x_vertex:torch.Tensor)->torch.Tensor:
        r"""
        Applies concatenation to edge feature and the difference between vertex feature and edge feature. This implementation is inspired from `Difference Residual Graph Neural Networks@ACMMM2022 <https://yangliang.github.io/pdf/mm22.pdf>`_

        .. math::
            \text{Concat}(x_{edge}, x_{vertex}) = \text{cat}([x_{edge}, x_{vertex}-x_{edge}]),
            
        Shape:
           - x_edge: :math:`(B, D_{edge}, N, M)`
           - x_vertex: :math:`(B, D_{vertex}, N, M)`
           - output:  :math:`(B, D_{edge}+D_{vertex}, N, M)`　 
        Args:
           x_edge: a batched tensor of edge-wise features
           x_vertex: a batched tensor of vertex-wise features

        Returns:
           x_merged
        """
        assert(self.dim_feature==-1)
        D_edge = x_edge.shape[self.dim_feature]
        D_vertex = x_vertex.shape[self.dim_feature]
        D_min = min(D_edge, D_vertex)
        shape = x_vertex.shape
        x_vertex = x_vertex.reshape(-1,D_vertex)
        x_vertex[:, :D_min] -= x_edge.view(-1, D_edge)[:, :D_min]
        x_vertex = x_vertex.view(shape)
        return super().forward(x_edge, x_vertex)

class ConcatMergerAny(nn.Module):   
    def __init__(self, dim_feature:int=-1):
        super().__init__()
        self.dim_feature = dim_feature
        
    def forward(self,
               xs:List[torch.Tensor])->torch.Tensor:
        r"""
        Applies concatenation to edge feature and vertex feature

        .. math::
            \text{Concat}(x_{edge}, x_{vertex}) = \text{cat}([x_{edge}, x_{vertex}]),
        Shape:
           - x_edge: :math:`(\ldots, N, M, D_{edge})`
           - x_src_vertex: :math:`(\ldots, N, M, D_{vertex})`
           - x_tar_vertex: :math:`(\ldots, N, M, D_{vertex})`
           - output:  :math:`(\ldots, N, M, D_{edge}+2*D_{vertex})`　 
        Args:
           x_edge: a batched tensor of edge-wise features
           x_src_vertex: a batched tensor of features for each source vertices.
           x_tar_vertex: a batched tensor of features for each target vertices.

        Returns:
           x_merged
        """
        return torch.cat(xs, dim=self.dim_feature)
    
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
                dim_target:int)->torch.Tensor:
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
        z_vertex = self.aggregator(z, dim_target)
        z = self.merger(x, z_vertex)
        return self.second_process(z)
        '''
        if not self.return_vertex_feature:
            return z
        return z, z_vertex
        '''

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
    
class SetEncoderPointNetCrossDirectional(SetEncoderBase):
    def __init__(self, in_channels:int, mid_channels:int, output_channels:int, **kwargs):
        r"""        
        Args:
            in_channels: the number of input channels.
            mid_channels: the number of output channels at the first convolution.
            out_channels: the number of output channels at the second convolution.
           
        """ 
        first_process = nn.Linear(in_channels, mid_channels)
        second_process = nn.Linear(in_channels + 2*mid_channels, output_channels, bias=False)    

        super().__init__(
            first_process, 
            MaxPoolingAggregator(),
            ConcatMergerAny(dim_feature=-1),
            second_process,
            **kwargs,
        )
    def forward(self, 
                x:torch.Tensor,
                dim_target:int)->torch.Tensor:
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
        if dim_target==-2:
            dim_src = -3
        elif dim_target==-3:
            dim_src = -2
        else:
            raise RuntimeError("Unexpected dim_tar: {}.".format(dim_target))
        z_src_vertex_fut = xab_fut = torch.jit.fork(self.aggregator, z, dim_src)
        z_tar_vertex = self.aggregator(z, dim_target)
        z_src_vertex = torch.jit.wait(z_src_vertex_fut)
        z = self.merger(x, z_src_vertex, z_tar_vertex)
        return self.second_process(z)
        '''
        if not self.return_vertex_feature:
            return z
        return z, z_vertex
        '''
class SetEncoderPointNetTotalDirectional(SetEncoderBase):
    def __init__(self, in_channels:int, mid_channels:int, output_channels:int, **kwargs):
        r"""        
        Args:
            in_channels: the number of input channels.
            mid_channels: the number of output channels at the first convolution.
            out_channels: the number of output channels at the second convolution.
           
        """ 
        first_process = nn.Linear(in_channels, mid_channels)
        second_process = nn.Linear(in_channels + 3*mid_channels, output_channels, bias=False)    

        super().__init__(
            first_process, 
            MaxPoolingAggregator(),
            ConcatMergerAny(dim_feature=-1),
            second_process,
            **kwargs,
        )
    def forward(self, 
                x:torch.Tensor,
                dim_target:int)->torch.Tensor:
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
        if dim_target==-2:
            dim_src = -3
        elif dim_target==-3:
            dim_src = -2
        else:
            raise RuntimeError("Unexpected dim_tar: {}.".format(dim_target))
        z_src_vertex_fut = xab_fut = torch.jit.fork(self.aggregator, z, dim_src)
        z_tar_vertex = self.aggregator(z, dim_target)
        z_src_vertex = torch.jit.wait(z_src_vertex_fut)
        z = self.merger(x, z_src_vertex, z_tar_vertex, z_src_vertex.max(z_tar_vertex))
        return self.second_process(z)
        '''
        if not self.return_vertex_feature:
            return z
        return z, z_vertex
        '''             
        
        
StreamAggregator = Callable[[torch.Tensor,Optional[torch.Tensor], bool],Tuple[torch.Tensor,torch.Tensor,torch.Tensor]]
class DualSoftmax(nn.Module):
    r"""
    
    Applies the dual-softmax calculation to a batched matrices. DualSoftMax is originally proposed in `LoFTR (CVPR2021) <https://zju3dv.github.io/loftr/>`_. 
    
    .. math::
        \text{DualSoftmax}(x^{ab}_{ij}, x^{ba}_{ij}) = \frac{\exp(x^{ab}_{ij})}{\sum_j \exp(x^{ab}_{ij})} * \frac{\exp(x^{ba}_{ij})}{\sum_i \exp(x^{ba}_{ij})} 

    In original definition, always :math:`x^{ba}=x^{ab}`. This is an extensional implementation that accepts :math:`x^{ba}\neq x^{ab}` to input the two stream outputs of `WeaveNet`. 
    
    """
    def __init__(self, dim_src:int=-3, dim_tar:int=-2)->None:
        super().__init__()
        self.sm_col = nn.Softmax(dim=dim_tar)
        self.sm_row = nn.Softmax(dim=dim_src)
        
    def apply_softmax(self,
                      xab:torch.Tensor, 
                      xba:Optional[torch.Tensor],
                      is_xba_transposed:bool,
                     )->Tuple[torch.Tensor, torch.Tensor]:
        if xba is None:
            xba_t = xab
        elif is_xba_transposed:
            xba_t = xba
        else:
            xba_t = xba.transpose(-3,-2)
        zab_fut = torch.jit.fork(self.sm_col, xab)
        zba = self.sm_row(xba_t)
        zab = torch.jit.wait(zab_fut)
        return zab, zba
    
    

    def forward(self, 
                xab:torch.Tensor, 
                xba:Optional[torch.Tensor]=None, 
                is_xba_transposed:bool=True)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        r""" Calculate the dual softmax for batched matrices.
                
        Shape:
           - xab: :math:`(B, \ldots, N_1, M_1, D)`
           - xba: :math:`(B, \ldots, N_2, M_2, D)`
           - output:  :math:`(B, \ldots, N_1, M_1, D)`
           
        Args:
           xab: 1st batched matrices.
           
           xba: 2nd batched matrices. x_ab is used as (not-transposed) x_ba if None. This option corresponds to the original implementation of LoFTR.
           
           is_xba_transposed: set False if :math:`(N_1, M_1)==(N_2, M_2)` and set True if :math:`(N_1, M_1)==(M_2, N_2). Default: False
           
        Returns:
           values mab * mba_t, mab (=softmax(xab, dim=-2)), mba_t (=softmax(xba_t, dim=-1)
           
        """
        zab, zba_t = self.apply_softmax(xab, xba, is_xba_transposed)
        return zab * zba_t, zab, zba_t

class DualSoftmaxSqrt(DualSoftmax):
    r"""
    
    A variation of :obj:`DualSoftMax` for evenly weighting backward values for two streams.
    
    .. math::
        \text{DualSoftmaxSqrt}(x^{ab}_{ij}, x^{ba}_{ij}) = \sqrt{\text{DualSoftmax}(x^{ab}_{ij}, x^{ba}_{ij})}
    
    """    
    def forward(self, 
                xab:torch.Tensor, 
                xba:Optional[torch.Tensor]=None, 
                is_xba_transposed:bool=True)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        epsilon:float=10**-7
        r""" Calculate the dual softmax for batched matrices.
        Shape:
           - x_ab: :math:`(B, \ldots, N_1, M_1, D)`
           - x_ba: :math:`(B, \ldots, N_2, M_2, D)`
           - output:  :math:`(B, \ldots, N_1, M_1, D)`
           
        Args:
            x_ab: 1st batched matrices
            
            x_ba: 2nd batched matrices. x_ab is used as (not-transposed) x_ba if None. This option corresponds to the original implementation of LoFTR.
            
            is_ba_transposed: True if rows1==cols2 and cols1==rows2. False if rows1 == rows2 and cols1 == cols2. Default: True
       Returns:
           values (mab * mba_t).sqrt(), mab (=softmax(xab, dim=-2)), mba_t (=softmax(xba_t, dim=-1)
        """
        zab, zba_t = self.apply_softmax(xab, xba, is_xba_transposed)
        return torch.clamp(zab*zba_t, epsilon).sqrt(), zab, zba_t

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
           - x_ab: :math:`(B, \ldots, N_1, M_1, D)`
           - x_ba: :math:`(B, \ldots, N_2, M_2, D)`
           - output:  :math:`(B, \ldots, N_1, M_1, D)`
           
        Args:
           x_ab: 1st batched matrices.
           
           x_ba: 2nd batched matrices. x_ab is used as (not-transposed) x_ba if None. This option corresponds to the original implementation of LoFTR.
           
           is_ba_transposed: set False if :math:`(N_1, M_1)==(N_2, M_2)` and set True if :math:`(N_1, M_1)==(M_2, N_2). Default: False
           
        Returns:
           values torch.min(mab, mba_t), mab (=softmax(xab, dim=-2)), mba_t (=softmax(xba_t, dim=-1)
           
        """
        zab, zba_t = self.apply_softmax(xab, xba, is_xba_transposed)
        return zab.min(zba_t), zab, zba_t
        
class CrossConcatVertexFeatures(Interactor):
    def __init__(self,
                 dim_a = -3,
                 dim_b = -2,
                 dim_feature = -1,
                 compute_similarity:Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 directional_normalization:Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                ):
        self.dim_a, self.dim_b, self.dim_feature = dim_a, dim_b, dim_feature
        self.compute_similarity = compute_similarity
        self.directional_normalization = directional_normalization
        
    def forward(self, xa:torch.Tensor, xb:torch.Tensor)->torch.Tensor:
        xa = xa.unsqueeze(dim = self.dim_b)
        xb = xb.unsqueeze(dim = self.dim_a)        
        shape = xa.shape
        shape[self.dim_b] = xb.size(self.dim_b)        
        
        if self.compute_similarity is None:
            return torch.cat([xa.expand(shape), xb.expand(shape)], dim=self.dim_feature)
        
        similarity_matrix = self.compute_similarity(xa, xb, dim=self.dim_feature)
        shape_sim = shape
        shape_sim[self.dim_feature] = 1
        
        if self.directional_normalization is None:
            return torch.cat([xa.expand(shape), xb.expand(shape), similarity_matrix.view(shape_sim)], dim=self.dim_feature)
        
        sim_a = self.directional_normalization(similarity_matrix, dim=self.dim_a)
        sim_b = self.directional_normalization(similarity_matrix, dim=self.dim_b)
        return  torch.cat([xa.expand(shape), sim_a.view(shape_sim), xb.expand(shape), sim_b.view(shape_sim)], dim=self.dim_feature)
        
    def output_channels(self, input_channels:int)->int:
        output_channels = 2 * input_channels
        if self.compute_similarity is None:
            return output_channels
        
        if self.directional_normalization is None:
            # output_channels = xa's channels + 1 + xb's channels 
            return output_channels + 1
        
        # output_channels = xa's channels + col-wisely-normalized similarity + xb's channels + row-wisely-normalized similarity 
        return output_channels + 2
    
    

        