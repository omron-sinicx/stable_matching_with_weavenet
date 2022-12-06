import torch
from torch import nn


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

    def forward(self,x,dim=2):
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
                       max_pool_concat(x[batch_size:],z[batch_size:],3)], dim=0)
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
    
class WeaveNet(torch.nn.Module):
    def __init__(self, L, D=64, inner_conv_out_channels=64, use_resnet=True, sab_dim=1, asymmetric=False, stream_aggregation='HV', solver=None):
        super(WeaveNet,self).__init__()
        self.sab_dim = sab_dim
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
        if stream_aggregation == 'HV':
            self.stream_aggregation = self.stream_aggregation_HV
        elif stream_aggregation == 'dual_softmax':
            self.stream_aggregation = self.stream_aggregation_dual_softmax
        elif stream_aggregation == 'linear_problem':
            self.lp_solver = solver
            self.stream_aggregation = self.stream_aggregation_linear_problem
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
            input_dim = (self.sab_dim+1)*2
        else:
            input_dim = self.sab_dim*2

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
    
    def stream_aggregation_linear_problem(self, m : torch.Tensor) -> torch.Tensor:
        m = self.stream_aggregation_dual_softmax(m) # bind m to be solvable.
        return solver(m)
    
    def forward(self, Ss):
        Za, Zb = Ss
        zshape = Za.shape
        batch_size = zshape[0]
        N = zshape[-2]
        M = zshape[-1]
        Zb = Zb.transpose(-2,-1)

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
        
        return self.stream_aggregation(m), m[0], m[1] # return m, mab, mba


if __name__ == "__main__":
    _ = SimpleDenseNet()
