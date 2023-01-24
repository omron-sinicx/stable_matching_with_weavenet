import torch
from torch import nn
from .preference import to_rank, PreferenceFormat
from .layers import *

from typing import List, Optional

try:
    # Literal is available with python >=3.8.0
    from typing import Literal
except:
    # pip install typing_extensions with python < 3.8.0
    from typing_extensions import Literal

class WeaveNet(nn.Module):
    def __init__(self,
                 head:nn.Module,
                 output_channels:int=1,
                stream_aggregator:Optional[StreamAggregator] = DualSoftmaxSqrt(dim_src=-3, dim_tar=-2)):
        super().__init__()
        self.head = head
        input_channels_last = head.interactor.output_channels(head.output_channels)
        self.last_layer = nn.Sequential(
            nn.Linear(input_channels_last, output_channels, bias=False),
            BatchNormXXC(output_channels),
        )
        self.stream_aggregator = stream_aggregator
    def forward(self, xab:torch.Tensor, xba_t:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xab, xba_t = self.head(xab, xba_t)
        xab, xba_t = self.head.interactor(xab, xba_t)
        xab = self.last_layer(xab)
        xba_t = self.last_layer(xba_t)
        m, mab, mba_t = self.stream_aggregator(xab, xba_t)
        return m, mab, mba_t
        
        
        
class TwoStreamGNNHead(nn.Module):
    def __init__(self,
                 modules: Tuple[nn.ModuleList, nn.ModuleList],
                 interactors: List[Interactor]=None,                 
                 calc_residual:Optional[List[bool]]=None,
                 keep_first_var_after:int=0,
                ):
        super().__init__()
        
        
        self.L = len(modules[0])
        assert(self.L == len(modules[1]))
        
        self.modules = list(modules) # store as a list to make `self.modules` rewritable 
        
        self.interactors = interactors
        assert(self.L == len(self.interactors))
        
        if calc_residual is None:
            self.calc_residual = [False] * self.L
            self.use_residual = False
        else:            
            assert(self.L == len(calc_residual))
            self.calc_residual = calc_residual
            self.use_residual = sum(self.calc_residual)>0
            self.keep_first_var_after = keep_first_var_after
            assert(0 == sum(self.calc_residual[:self.keep_first_var_after]))
                    
        
    def forward(self, xab:torch.Tensor, xba_t:torch.Tensor)->torch.Tensor:
        xab_keep, xba_t_keep = None, None
        for l, (interactor, module0, module1, calc_res) in enumerate(zip(self.interactors, self.modules[0],self.modules[1], self.calc_residual)):
            xab, xba_t = interactor(xab, xba_t)            
            xab = module0(xab, dim_target=-2)
            xba_t = module1(xba_t, dim_target=-3)
            
            if not self.use_residual:
                continue
                
            if i==self.keep_first_var_after:
                # keep values after the first module process.
                xab_keep, xba_t_keep = xab, xba_t
                
            if calc_res:
                xab_keep, xab = xab, xab + xab_keep
                xba_t_keep, xba = xba_t, xba_t + xba_t_keep
        
        return xab, xba_t

SubblockProcOrder = Literal['ena','nae','ean','ane']
class SubBlock(nn.Module):
    def __init__(self, 
                 encoder:nn.Module, 
                 order: SubblockProcOrder,
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
    

class WeaveNetHead(TwoStreamGNNHead):
    interactor = CrossConcat(dim_feature=-1)
    def __init__(self,
                 input_channel:int,
                 output_channels:List[int],
                 mid_channels:List[int],
                 first_interactor:nn.Module=CrossConcat(dim_feature=-1),
                 *args,
                 **kwargs):
        

        # set in_channels at each layer
        in_chs = [input_channel]+[self.calc_in_channel(out_ch) for out_ch in output_channels[:-1] ]
        
        # L: the number of layers
        L = len(in_chs)
        
        assert(L==len(output_channels))
        assert(L==len(mid_channels))
            
        modules:List[SubBlock] = [self.build_module(in_ch, mid_ch, out_ch) for in_ch, mid_ch, out_ch in zip(in_chs, mid_channels, output_channels)]
        
        
        interactors:List[CrossConcat] = [first_interactor] + [self.interactor for l in range(1,L)]
        super().__init__(
            (modules, modules),
            interactors,       
            *args,
            **kwargs,
        )
        self.weavenet_modules = nn.ModuleList(modules)
        self.output_channels = output_channels[-1]

    def calc_in_channel(self, prev_out:int)->int:
        return 2 * prev_out
            
    def build_module(self, in_ch:int, mid_ch:int, out_ch:int)->nn.Module:
        activation = nn.PReLU()
        return SubBlock(
            SetEncoderPointNet(in_ch, mid_ch, out_ch),
            'ena', # encoder -> normalization -> activation
            BatchNormXXC(out_ch),
            activation,
        )
    
class WeaveNetPlusHead(WeaveNetHead):
    
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

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
    def calc_in_channel(self, prev_out:int)->int:
        return 2*prev_out

            
    def build_module(self, in_ch:int, mid_ch:int, out_ch:int)->nn.Module:
        activation = nn.PReLU()
        return SubBlock(
            self.Encoder(in_ch, mid_ch, out_ch),
            'ena', # encoder -> normalization -> activation
            BatchNormXXC(out_ch),
            activation,           
        )
    


class WeaveNetBatchNormSplitHead(WeaveNetHead):
    def __init__(self, input_channel:int,
                 output_channels:List[int],
                 mid_channels:List[int],
                 *args, **kwargs):
        super().__init__(input_channel,
                         output_channels,
                         mid_channels,
                         *args, **kwargs)
        
        # register only batchnorms as a sub-module since module2 are mostly a shallow copy of self.module[0]
        self.batchnorms4stream2 = nn.ModuleList([nn.BatchNorm2D(out_ch) for out_ch in output_channels])
        modules:List[SubBlock] = [
            SubBlock(
                self.module[0][i].encoder,
                self.module[0][i].order,
                self.module[0][i].normalizer,
                self.module[0][i].activator,
            ) for i in range(self.L)
        ]    
        self.modules[1] = modules2
        self.weavenet_modules2 = nn.ModuleList(self.modules[1])

        
class WeaveNetDualHead(WeaveNetHead):
    def __init__(self, input_channel:int,
                 output_channels:List[int],
                 mid_channels:List[int],
                 *args, **kwargs):
        super().__init__(input_channel,
                         output_channels,
                         mid_channels,
                         *args, **kwargs)
        
        # build another modules for the 2nd stream.
        self.modules[1] = [self.build_module(in_ch, mid_ch, out_ch) for in_ch, mid_ch, out_ch in zip(in_chs, mid_channels, output_channels)]

class SingleStreamGNNHead(nn.Module):
    def __init__(self,
                 modules: nn.ModuleList,
                 interactor: Optional[Interactor]=None,                 
                 calc_residual:Optional[List[bool]]=None,
                 keep_first_var_after:int=0,
                ):
        super().__init__()
        
        
        self.L = len(modules)
        
        self.modules = list(modules) # store as a list to make `self.modules` rewritable 
        
        self.interactor = interactor
        
        if calc_residual is None:
            self.use_residual = False
        else:            
            assert(self.L == len(calc_residual))
            self.calc_residual = calc_residual
            self.use_residual = sum(self.calc_residual)>0
            self.keep_first_var_after = keep_first_var_after
            assert(0 == sum(self.calc_residual[:self.keep_first_var_after]))
            
        
        
    def forward(self, xab:torch.Tensor, xba_t:torch.Tensor)->torch.Tensor:
        x_keep = None
        x = self.interactor(xab, xba_t)
        for l, module in enumerate(self.modules):
            if l%2==0:
                dim = -2
            else:
                dim = -3
            x = module(x, dim_target=dim)
            
            if not self.use_residual:
                continue
                
            if i==self.keep_first_var_after:
                # keep values after the first module process.
                x_keep = x
                
            if self.calc_residual[l]:
                x_keep, x = x, x + x_keep
        return x
    
class WeaveNetSingleStreamHead(SingleStreamGNNHead):
    def __init__(self,
                 input_channel:int,
                 output_channels:List[int],
                 mid_channels:List[int],
                 *args,
                 **kwargs):
        
        
        # set in_channels at each layer
        in_chs = [input_channel]+[2*out_ch for out_ch in output_channels[:-1] ]
        
        # L: the number of layers
        L = len(in_chs)
        
        assert(L==len(output_channels))
        assert(L==len(mid_channels))
        
        modules:List[SubBlock] = [self.build_module(in_ch, mid_ch, out_ch) for in_ch, mid_ch, out_ch in zip(in_chs, mid_channels, output_channels)]
        
        super().__init__(
            modules,
            CrossConcat(),       
            *args,
            **kwargs,
        )
        self.weavenet_modules = nn.ModuleList(modules)
    

            
    def build_module(self, in_ch:int, mid_ch:int, out_ch:int)->nn.Module:
        activation = nn.PReLU()
        module = SubBlock(
            SetEncoderPointNet(in_ch, mid_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            activation,           
        )
        return module
    

WeaveNetType = Literal['standard', 'batchnorm split', 'dual', 'plus']        
class WeaveNetHead6(nn.Module):
    def __init__(self, 
                 input_channels:int, 
                 mid_channels:int = 64,
                 output_channels:int = 32,
                 mode: WeaveNetType='standard',
                ):
        super().__init__()
        if mode == 'standard':
            self.backbone = WeaveNetHead
        elif mode == 'batchnorm split':
            self.backbone = WeaveNetBatchNormSplitHead
        elif mode == 'dual':
            self.backbone = WeaveNetDualHead
        elif mode == 'plus':
            self.backbone = WeaveNetPlusHead
        else:
            raise NotImplementedError()
        self.build(input_channels, mid_channels, output_channels)
        
    def build(self, 
                 input_channels:int, 
                 mid_channels:int = 64,
                 output_channels:int = 32,
             )->None:
        self.module = self.backbone(
            input_channel,
            [output_channels]*6,
            [mid_channels]*6,
            calc_residual = None,
            keep_first_var_after=0,
        )
        
    def forward(self, xab, xba_t):
        return self.module(xab, xba_t)
        
class WeaveNetHead18(WeaveNetHead6):
    def build(self, 
                 input_channels:int=2, 
                 mid_channels:int=64,
                 output_channels:int=32,
             )->None:
        super().build()
        
        self.module2 = self.backbone(
            64,
            [mid_channels]*12,
            [output_channels]*12,
            calc_residual = [0,0,1]*4,
            keep_first_var_after=0,
        )
        
    def forward(self, xab, xba_t):
        xab, xba_t = self.module(xab, xba_t)            
        return self.module2(xab, xba_t)
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
                 
