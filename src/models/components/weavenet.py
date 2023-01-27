import torch
from torch import nn

from .preference import to_rank, PreferenceFormat
from .layers import *
from copy import deepcopy

from typing import List, Optional
from collections import UserList

try:
    # Literal is available with python >=3.8.0
    from typing import Literal
except:
    # pip install typing_extensions with python < 3.8.0
    from typing_extensions import Literal

class TrainableMatchingModule(nn.Module):
    def __init__(self,
                 head:nn.Module,
                 output_channels:int=1,
                 pre_interactor:Optional[CrossConcat] = CrossConcat(),
                 stream_aggregator:Optional[StreamAggregator] = DualSoftmaxSqrt(dim_src=-3, dim_tar=-2)):
        super().__init__()
        self.pre_interactor = pre_interactor
        self.head = head
        self.last_layer = nn.Sequential(
            nn.Linear(head.output_channels, output_channels, bias=False),
            BatchNormXXC(output_channels),
        )
        self.stream_aggregator = stream_aggregator
        
    def forward(self, xab:torch.Tensor, xba_t:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # first interaction
        if self.pre_interactor is not None:
            xab, xba_t = self.pre_interactor(xab, xba_t)
            
        # head
        xab, xba_t = self.head.forward(xab, xba_t)

        # wrap up into logits
        xab = self.last_layer(xab)
        xba_t = self.last_layer(xba_t)
        
        # aggregate two streams while applying logistic regression.
        m, mab, mba_t = self.stream_aggregator(xab, xba_t)
        return m, mab, mba_t

UnitProcOrder = Literal['ena','nae','ean','ane']
class Unit(nn.Module):
    def __init__(self, 
                 encoder:nn.Module, 
                 order: UnitProcOrder,
                 normalizer:Optional[nn.Module]=None, 
                 activator:Optional[nn.Module]=None):
        super().__init__()
        self.encoder = encoder
        self.normalizer = normalizer
        self.activator = activator
        self.order = order
        self.forward = eval("self.forward_{}".format(order))
        
    def forward_ena(self, x:torch.Tensor, dim_target:int)->torch.Tensor:
        x = self.encoder(x, dim_target)
        if self.normalizer is not None:
            x = self.normalizer(x)
        if self.activator is not None:
            x = self.activator(x)
        return x
    def forward_nae(self, x:torch.Tensor, dim_target:int)->torch.Tensor:
        if self.normalizer is not None:
            x = self.normalizer(x)
        if self.activator is not None:
            x = self.activator(x)
        x = self.encoder(x, dim_target)
        return x
    
    def forward_ean(self, x:torch.Tensor, dim_target:int)->torch.Tensor:
        x = self.encoder(x, dim_tar)
        if self.activator is not None:
            x = self.activator(x)
        if self.normalizer is not None:
            x = self.normalizer(x)
        return x
    
    def forward_ane(self, x:torch.Tensor, dim_target:int)->torch.Tensor:
        if self.activator is not None:
            x = self.activator(x)
        if self.normalizer is not None:
            x = self.normalizer(x)
        x = self.encoder(x, dim_target)
        return x
    
    def forward(self, x:torch.Tensor, dim_target:int)->torch.Tensor:
        return x    
    
class UnitList(UserList):
    def __init__(self,
                 input_channels:int,
                 output_channels_list:List[int],
                ):
        self.input_channels = input_channels
        self.output_channels_list = output_channels_list    
        
    def build(self, interactor:Optional[Interactor]=None):            
            
        if interactor:
            in_chs = [self.input_channels]+[interactor.output_channels(out_ch) for out_ch in self.output_channels_list[:-1] ]
        else:
            in_chs = [self.input_channels]+[out_ch for out_ch in self.output_channels_list[:-1] ]
            
        L = len(in_chs)
        self._build(in_chs)
                
    def _build(self, in_channels_list:List[int]):     
        self.data = [] # this parent class behaves as an empty Unit list.
        
ExclusiveElementsOfUnit = Literal['none', 'normalizer', 'all'] # standard, biased, dual
class MatchingModuleHead(nn.Module):
    def __init__(self,
                 module_units: UnitList,
                 interactor:Optional[Interactor]=None,
                 calc_residual:Optional[List[bool]]=None,
                 keep_first_var_after:int=0,
                 exclusive_elements_of_unit:ExclusiveElementsOfUnit='none',
                ):
        super().__init__()
        if interactor:
            self.interactor = interactor
        module_units.build(interactor)
        # prepare for residual paths.
        L = len(module_units)
        if calc_residual is None:
            self.calc_residual = [False] * L
            self.use_residual = False
        else:            
            assert(L == len(calc_residual))
            self.calc_residual = calc_residual
            self.use_residual = sum(self.calc_residual)>0
            self.keep_first_var_after = keep_first_var_after
            assert(0 == sum(self.calc_residual[:self.keep_first_var_after]))
            
        self.build_two_stream_structure(module_units, exclusive_elements_of_unit, interactor is None)
        
        self.input_channels = module_units.input_channels
        
        if interactor:
            self.output_channels = self.interactor.output_channels(module_units.output_channels_list[-1])
        else:
            self.output_channels = module_units.output_channels_list[-1]

        
    def build_two_stream_structure(self, 
                                    module_units: List[nn.Module],
                                    exclusive_elements_of_unit:ExclusiveElementsOfUnit='none',
                                    is_single_stream:bool = False,
                                   )->None:
        # register module_units as a child nn.Module.
        self.__units = nn.ModuleList(module_units)
        
        if is_single_stream:
            # !!!override forward by forward_single_stream!!!
            assert(exclusive_elements_of_unit=='none') 
            # assert non-default exclusive_... value for the fail-safe (the value is ignored when is_single_stream==True).
            self.forward = self.forward_single_stream
            self.stream = module_units
        # make 2nd stream
        elif exclusive_elements_of_unit == 'none':
            self.streams = [module_units, module_units] # shallow copy
            return
        elif exclusive_elements_of_unit == 'all':
            module_units2 = deepcopy(module_units) # deep copy
            self.streams = [module_units, module_units2]
            self.__units2 = nn.ModuleList(module_units2)
            return
        elif exclusive_elements_of_unit == 'normalizer':
            module_normalizers2 = [deepcopy(m.normalizer) for m in module_units]  # deep copy normalizers   
            self.__units2 = nn.ModuleList(module_normalizers2)
            module_units2:List[ModuleUnits] = [                
                ModuleUnit(
                    unit.encoder, # shallow copy
                    unit.order, # shallow copy
                    normalizer, # deep-copied normalizer 
                    unit.activator, # shallow copy
                )
                for unit, normalizer in zip(module_units, module_normalizers2)
            ]
            self.streams = [module_units, module_units2]
            return
        else:
            raise RuntimeError("Unknown ExclusiveElementsOfUnit: {}".format(exclusive_elements_of_unit))
            
            
    def forward(self, xab:torch.Tensor, xba_t:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        xab_keep, xba_t_keep = None, None
        for i, (unit0, unit1, calc_res) in enumerate(zip(self.streams[0],self.streams[1], self.calc_residual)):
            xab = unit0(xab, dim_target=-2)
            xba_t = unit1(xba_t, dim_target=-3)
            
            if self.use_residual:              
                if i==self.keep_first_var_after:
                    # keep values after the directed unit's process.
                    xab_keep = xab
                    xba_t_keep = xba_t
                if calc_res:
                    xab_keep, xab = xab, xab + xab_keep
                    xba_t_keep, xba = xba_t, xba_t + xba_t_keep
            
            xab, xba_t = self.interactor(xab, xba_t)            
        
        return xab, xba_t
    
    def forward_single_stream(self, xab:torch.Tensor, xba_t:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        x_keep = None
        for l, (unit, calc_res) in enumerate(zip(self.stream, self.calc_residual)):
            if l%2==0:
                dim = -3
            else:
                dim = -2
            xab = unit(xab, dim_target=dim)
            
            if self.use_residual:              
                if i==self.keep_first_var_after:
                    # keep values after the directed unit's process.
                    xab_keep = xab
                if calc_res:
                    xab_keep, xab = xab, xab + xab_keep            
        return xab, xab
    


        
class WeaveNetUnitList(UnitList):
    def __init__(self,
                 input_channels:int,
                 output_channels_list:List[int],
                 mid_channels_list:List[int],
            ):
        self.mid_channels_list = mid_channels_list
        super().__init__(input_channels, output_channels_list)
        assert(len(output_channels_list) == len(mid_channels_list))               

        
    def _build(self, in_channels_list:List[int]):
        self.data = [
            Unit(
                SetEncoderPointNet(in_ch, mid_ch, out_ch),
                'ena',
                BatchNormXXC(out_ch),
                nn.PReLU(),)
            for in_ch, mid_ch, out_ch in zip(in_channels_list, self.mid_channels_list, self.output_channels_list)
        ]
        
class WeaveNetExperimentalUnitList(WeaveNetUnitList):
    class Encoder(SetEncoderBase):
        def __init__(self, in_channels:int, mid_channels:int, output_channels:int, **kwargs):
            r"""        
            Args:
                in_channels: the number of input channels.
                mid_channels: the number of output channels at the first convolution.
                output_channels: the number of output channels at the second convolution.

            """ 
            first_process = nn.Linear(in_channels, mid_channels)
            second_process = nn.Linear(in_channels + mid_channels, output_channels, bias=False)    

            super().__init__(
                first_process, 
                MaxPoolingAggregator(),
                DifferenceConcatMerger(dim_feature=-1),
                second_process,
                **kwargs,
            )
    def _build(self, in_channels_list:List[int]):
        self.data = [
            Unit(
                self.Encoder(in_ch, mid_ch, out_ch),
                'ena',
                BatchNormXXC(out_ch),
                nn.PReLU(),)
            for in_ch, mid_ch, out_ch in zip(in_channels_list, self.mid_channels_list, self.out_channels_list)
        ]                 
        
class WeaveNetHead(MatchingModuleHead):
    def __init__(self,
                 input_channels:int,
                 output_channels_list:List[int],
                 mid_channels_list:List[int],
                 calc_residual:Optional[List[bool]]=None,
                 keep_first_var_after:int=0,
                 exclusive_elements_of_unit:ExclusiveElementsOfUnit='none',
                 is_single_stream:bool=False,
                ):
        if is_single_stream:
            interactor = None
        else:
            interactor = CrossConcat()
            
        super().__init__(
            WeaveNetUnitList(input_channels, output_channels_list, mid_channels_list),
            interactor = interactor,            
            calc_residual = calc_residual,
            keep_first_var_after = keep_first_var_after,
            exclusive_elements_of_unit = exclusive_elements_of_unit,
        )        
        
class WeaveNetExperimentalHead(MatchingModuleHead):
    def __init__(self,
                 input_channels:int,
                 output_channels:List[int],
                 mid_channels:List[int],
                 calc_residual:Optional[List[bool]]=None,
                 keep_first_var_after:int=0,
                 exclusive_elements_of_unit:ExclusiveElementsOfUnit='none',
                ):
        super().__init__(
            WeaveNetExperimentalUnitList(input_channels, output_channels_list, mid_channels_list),
            interactor = CrossConcat(),            
            calc_residual = calc_residual,
            keep_first_var_after = keep_first_var_after,
            exclusive_elements_of_unit = exclusive_elements_of_unit,
        )        
        
"""
@torch.jit.script
def max_pool_concat(x,z,dim:int):
    z_max = z.max(dim,keepdim=True)[0]
    z = z_max.expand(z.shape)
    return torch.cat([x,z],dim=1)

class StreamSelectiveLayer(nn.Module):
    def __init__(self,n_streams,base_layer,*args,**kwargs):
        super(StreamSelectiveLayer,self).__init__()
        self.layers = nn.ModuleList()
        self.layers += [base_layer(*args,**kwargs) for i in range(n_streams)]

    def forward(self,x,str_id=0):
        return self.layers[str_id](x)
    
class EncoderMaxPool(nn.Module):
    def __init__(self,input_dim,output_dim,output_dim_max=None,use_batch_norm=True,activation=nn.PReLU(), asymmetric=False):
        super(EncoderMaxPool,self).__init__()
        if output_dim_max is None:
            output_dim_max = output_dim * 3

        self.conv_max = nn.Conv2d(input_dim, output_dim_max, kernel_size=1)
        self.conv = nn.Conv2d(input_dim+output_dim_max, output_dim, kernel_size=1,bias=False)

        self.asymmetric = asymmetric
        if use_batch_norm:
            if asymmetric:
                self.bn = StreamSelectiveLayer(2,nn.BatchNorm2d,output_dim)
            else:
                self.bn = nn.BatchNorm2d(output_dim)
        else:
            self.bn = None
        self.act = activation

    def forward(self,x,dim=2): #dim =2 is false implementation???
        # dim = 2 or 3.
        z = self.conv_max(x)
        z = max_pool_concat(x,z,dim)
        z = self.conv(z)
        if self.bn is not None:
            if self.asymmetric:
                z = self.bn(z,dim-2)
            else:
                z = self.bn(z)
        return self.act(z)
    
class EncoderMaxPool2stream(EncoderMaxPool):
    def __init__(self,*args,**kwargs):
        super(EncoderMaxPool2stream,self).__init__(*args,**kwargs)

    def forward(self,x,batch_size:int):
        # dim = 2 or 3.
        z = self.conv_max(x)
        z = torch.cat([max_pool_concat(x[:batch_size],z[:batch_size],2),
                       max_pool_concat(x[batch_size:],z[batch_size:],3)], dim=0) # Bug. 論文に掲載されている実装は 3 -> 2のはず．しかし，3の方が性能が高い．
        z = self.conv(z)
        if self.bn is not None:
            if self.asymmetric:
                z = torch.cat([self.bn(z[:batch_size],0),
                               self.bn(z[batch_size:],1)],dim=0)
            else:
                z = self.bn(z)
        return self.act(z)    
    
@torch.jit.script
def cross_concat(Z,batch_size:int):
    Za = Z[:batch_size]
    Zb = Z[batch_size:]
    return torch.cat([torch.cat([Za,Zb],dim=1), torch.cat([Zb,Za],dim=1)],dim=0)


class FeatureWeavingLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, inner_dim,
                 encoder=EncoderMaxPool2stream, use_batch_norm=True,
                 activation=nn.PReLU(), asymmetric=False):

        super(FeatureWeavingLayer,self).__init__()
        self.E = encoder(input_dim,output_dim,inner_dim,use_batch_norm,activation,asymmetric)

    def forward(self,Z):
        batch_size = Z.shape[0]//2
        Z = cross_concat(Z,batch_size)
        return self.E(Z,batch_size)
    
class WeaveNetOldImplementation(torch.nn.Module):
    def __init__(self, L, D, inner_conv_out_channels, 
                 use_resnet=False, input_side_channels=1, asymmetric=False, stream_aggregation='dual_softmax_sqrt'):
        super().__init__()
        self.input_side_channels = input_side_channels
        self.use_resnet=use_resnet
        self.asymmetric=asymmetric
        self.L = L
        if hasattr(D, '__len__'):
            self.D = D
        else:
            assert(L>0)
            self.D = [D] * L #[D, D, ..., D] with length L.
            
        if hasattr(inner_conv_out_channels, '__len__'):
            self.inner_conv_out_channels = inner_conv_out_channels        
        else:
            assert(L>0)
            self.inner_conv_out_channels = [inner_conv_out_channels] * L #[D, D, ..., D] with length L.

        assert(len(self.D) == len(self.inner_conv_out_channels))

        self.build()
        
        self.softmax_ab = nn.Softmax(dim=-1)
        self.softmax_ba = nn.Softmax(dim=-2)
        if stream_aggregation is None:
            self.stream_aggregation = None
        if stream_aggregation == 'HV':
            self.stream_aggregation = self.stream_aggregation_HV
        elif stream_aggregation == 'dual_softmax':
            self.stream_aggregation = self.stream_aggregation_dual_softmax
        elif stream_aggregation == 'dual_softmax_sqrt':
            self.stream_aggregation = self.stream_aggregation_dual_softmax_sqrt
        else:
            raise RuntimeError("Unknown stream_aggregation: {}.".format(stream_aggregation))
            

    def build(self):
        self.encoders, dim = self.build_encoders()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1,bias=False),
            nn.BatchNorm2d(1)
        )

    def build_encoders(self):
        encoders = torch.nn.ModuleList()

        if self.asymmetric:
            input_dim = (self.input_side_channels+1)*2
        else:
            input_dim = self.input_side_channels*2

        for i in range(self.L):
            output_dim = self.D[i]
            assert(output_dim%2==0)            
            encoders.append(
                FeatureWeavingLayer(input_dim,output_dim//2,
                                    self.inner_conv_out_channels[i],
                                    EncoderMaxPool2stream,
                                    asymmetric=self.asymmetric)
            )
            input_dim = output_dim
        return encoders, input_dim

    def stream_aggregation_HV(self, m : torch.Tensor) -> torch.Tensor:
        m = (m[0]+m[1])/2.
        mab = self.softmax_ab(m)
        mba = self.softmax_ba(m)
        return torch.stack([mab,mba]).min(dim=0)[0]
    
    def stream_aggregation_dual_softmax(self, m : torch.Tensor) -> torch.Tensor:
        mab = self.softmax_ab(m[0])
        mba = self.softmax_ba(m[1])
        return mab * mba
    
    def stream_aggregation_dual_softmax_sqrt(self, m : torch.Tensor) -> torch.Tensor:
        mab = self.softmax_ab(m[0])
        mba = self.softmax_ba(m[1])
        return (mab * mba).sqrt()
    
    def forward(self, Za, Zb):
        Za = Za.permute(0,3,1,2)
        Zb = Zb.permute(0,3,1,2)
        Ss = [Za, Zb]
        zshape = Za.shape
        batch_size = zshape[0]
        N = zshape[-2]
        M = zshape[-1]
        #Zb = Zb.transpose(-2,-1)

        if self.asymmetric:
            condition = Za.new_zeros(batch_size,1,N,M) # 0
            Za = torch.cat([Za,condition], dim=1)
            condition += 1 # 1
            Zb = torch.cat([Zb,condition], dim=1)
        Z = torch.cat([Za,Zb],dim=0)
        Z_keep = None


        for i, FWLayer in enumerate(self.encoders):
            Z = FWLayer(Z)
            if self.use_resnet and i%2==0:
                # use residual network
                if Z_keep is not None:
                    Z = Z + Z_keep
                Z_keep = Z

        Z = self.conv1x1(cross_concat(Z,batch_size))
        
        m = torch.stack([Z[:batch_size].view(batch_size, N,M),Z[batch_size:].view(batch_size, N,M)])
        if self.stream_aggregation:
            return self.stream_aggregation(m), m[0], m[1] # return m, mab, mba
        
        return None, m[0], m[1] # return only mab, mba if stream_aggregation is None.    
"""
if __name__ == "__main__":
    #_ = WeaveNetOldImplementation(2, 2,1)
    _ = WeaveNet(
            WeaveNetHead6(1,), 2, #input_channel:int,
                 [4,8,16], #out_channels:List[int],
                 [2,4,8], #mid_channels:List[int],1,2,2)
                 calc_residual=[False, False, True],
                 keep_first_var_after = 0,
                 stream_aggregator = DualSoftMaxSqrt())
                 
