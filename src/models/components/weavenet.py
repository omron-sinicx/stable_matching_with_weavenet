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
    
class UnitListGenerator():
    def __init__(self,
                 input_channels:int,
                 output_channels_list:List[int],
                ):
        self.input_channels = input_channels
        self.output_channels_list = output_channels_list    
        
    def generate(self, interactor:Optional[Interactor]=None):            
            
        if interactor:
            in_chs = [self.input_channels]+[interactor.output_channels(out_ch) for out_ch in self.output_channels_list[:-1] ]
        else:
            in_chs = [self.input_channels]+[out_ch for out_ch in self.output_channels_list[:-1] ]
            
        L = len(in_chs)
        return self._build(in_chs)
                
    def _build(self, in_channels_list:List[int]):     
        return [] # this parent class behaves as an empty Unit list.
        
ExclusiveElementsOfUnit = Literal['none', 'normalizer', 'all'] # standard, biased, dual
class MatchingModuleHead(nn.Module):
    def __init__(self,
                 units_generator: UnitListGenerator,
                 interactor:Optional[Interactor]=None,
                 calc_residual:Optional[List[bool]]=None,
                 keep_first_var_after:int=0,
                 exclusive_elements_of_unit:ExclusiveElementsOfUnit='none',
                ):
        super().__init__()
        if interactor:
            self.interactor = interactor
        units = units_generator.generate(interactor)
        # prepare for residual paths.
        L = len(units)
        self.keep_first_var_after = keep_first_var_after
        if calc_residual is None:
            self.calc_residual = [False] * L
            self.use_residual = False
        else:            
            assert(L == len(calc_residual))
            self.calc_residual = calc_residual
            self.use_residual = sum(self.calc_residual)>0
            assert(0 == sum(self.calc_residual[:self.keep_first_var_after]))
            
        self.build_two_stream_structure(units, exclusive_elements_of_unit, interactor is None)
        
        self.input_channels = units_generator.input_channels
        
        if interactor:
            self.output_channels = self.interactor.output_channels(units_generator.output_channels_list[-1])
        else:
            self.output_channels = units_generator.output_channels_list[-1]

        
    def build_two_stream_structure(self, 
                                    units:List[Unit],
                                    exclusive_elements_of_unit:ExclusiveElementsOfUnit='none',
                                    is_single_stream:bool = False,
                                   )->None:
        # register module_units as a child nn.Module.
        
        if is_single_stream:
            # !!!override forward by forward_single_stream!!!
            assert(exclusive_elements_of_unit=='none') 
            # assert non-default exclusive_... value for the fail-safe (the value is ignored when is_single_stream==True).
            self.forward = self.forward_single_stream
            self.stream = nn.ModuleList(units)
        # make 2nd stream
        elif exclusive_elements_of_unit == 'none':
            self.stream0 = nn.ModuleList(units)
            self.stream1 = nn.ModuleList(units) # shallow copy
            return
        elif exclusive_elements_of_unit == 'all':
            units2 = deepcopy(units) # deep copy
            self.stream0 = nn.ModuleList(units)
            self.stream1 = nn.ModuleList(units2) # shallow copy
            return
        elif exclusive_elements_of_unit == 'normalizer':
            module_normalizers2 = [deepcopy(m.normalizer) for m in units]  # deep copy normalizers   
            units2 = [                
                Unit(
                    unit.encoder, # shallow copy
                    unit.order, # shallow copy
                    normalizer, # deep-copied normalizer 
                    unit.activator, # shallow copy
                )
                for unit, normalizer in zip(module_units, module_normalizers2)
            ]
            self.stream0 = nn.ModuleList(units)
            self.stream1 = nn.ModuleList(units) # shallow copy
            return
        else:
            raise RuntimeError("Unknown ExclusiveElementsOfUnit: {}".format(exclusive_elements_of_unit))
            
            
    def forward(self, xab:torch.Tensor, xba_t:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        xab_keep, xba_t_keep = xab, xba_t
        for l, (unit0, unit1) in enumerate(zip(self.stream0, self.stream1)):
            calc_res = self.calc_residual[l]
            xab_fut = torch.jit.fork(unit0, xab, dim_target=-2)
            xba_t = unit1(xba_t, dim_target=-3)
            xab = torch.jit.wait(xab_fut)
            if self.use_residual:              
                if l==self.keep_first_var_after:
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
        for l, unit in enumerate(self.stream):
            calc_res = self.calc_residual[l]
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
    


        
class WeaveNetUnitListGenerator(UnitListGenerator):
    def __init__(self,
                 input_channels:int,
                 output_channels_list:List[int],
                 mid_channels_list:List[int],
            ):
        self.mid_channels_list = mid_channels_list
        super().__init__(input_channels, output_channels_list)
        assert(len(output_channels_list) == len(mid_channels_list))               

        
    def _build(self, in_channels_list:List[int]):
        return [
            Unit(
                SetEncoderPointNet(in_ch, mid_ch, out_ch),
                'ena',
                BatchNormXXC(out_ch),
                nn.PReLU(),)
            for in_ch, mid_ch, out_ch in zip(in_channels_list, self.mid_channels_list, self.output_channels_list)
        ]
        
class WeaveNetExperimentalUnitListGenerator(WeaveNetUnitListGenerator):
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
        return [
            Unit(
                #self.Encoder(in_ch, mid_ch, out_ch),
                SetEncoderPointNetTotalDirectional(in_ch, mid_ch, out_ch),
                'ena',
                BatchNormXXC(out_ch),
                nn.PReLU(),)
            for in_ch, mid_ch, out_ch in zip(in_channels_list, self.mid_channels_list, self.output_channels_list)
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
            WeaveNetUnitListGenerator(input_channels, output_channels_list, mid_channels_list),
            interactor = interactor,            
            calc_residual = calc_residual,
            keep_first_var_after = keep_first_var_after,
            exclusive_elements_of_unit = exclusive_elements_of_unit,
        )        
        
class WeaveNetExperimentalHead(MatchingModuleHead):
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
            WeaveNetExperimentalUnitListGenerator(input_channels, output_channels_list, mid_channels_list),
            interactor = interactor,            
            calc_residual = calc_residual,
            keep_first_var_after = keep_first_var_after,
            exclusive_elements_of_unit = exclusive_elements_of_unit,
        )        
        

if __name__ == "__main__":
    #_ = WeaveNetOldImplementation(2, 2,1)
    _ = WeaveNet(
            WeaveNetHead6(1,), 2, #input_channel:int,
                 [4,8,16], #out_channels:List[int],
                 [2,4,8], #mid_channels:List[int],1,2,2)
                 calc_residual=[False, False, True],
                 keep_first_var_after = 0,
                 stream_aggregator = DualSoftMaxSqrt())
                 
