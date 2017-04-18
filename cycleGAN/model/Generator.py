import torch.nn as nn
import torch

# courtesy: https://github.com/darkstar112358/fast-neural-style/blob/master/neural_style/transformer_net.py
class InstanceNormalization(torch.nn.Module):
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """

    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out

def build_conv_block(dim):
    return nn.Sequential(nn.ReflectionPad2d((1,1,1,1)),
                         nn.Conv2d(dim,dim,3,1,0),
                         InstanceNormalization(dim),
                         nn.ReLU(True),
                         nn.ReflectionPad2d((1,1,1,1)),
                         nn.Conv2d(dim,dim,3,1,0),
                         InstanceNormalization(dim))

class ResidualBlock(nn.Module):
    '''
    A single Residual Block:
    in ->
        conv(in_channels,out_channels,stride) -> BN -> ReLU
        conv(out_channels,out_channels,1) -> BN
    -> out
    (downsample)in + out
    '''
    def __init__(self, dim):
        super(ResidualBlock,self).__init__()
        self.conv = build_conv_block(dim)

    def forward(self, x):
        out = self.conv(x)
        out = out + x

        return out

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Generator,self).__init__()
        # 128 x 128
        self.layer1 = nn.Sequential(nn.ReflectionPad2d((3,3,3,3)),
                                    nn.Conv2d(input_nc,ngf,kernel_size=7,stride=1),
                                    InstanceNormalization(ngf),
                                    nn.ReLU(True))
        # 128 x 128
        self.layer2 = nn.Sequential(nn.Conv2d(ngf,ngf*2,kernel_size=3,stride=2,padding=1),
                                   InstanceNormalization(ngf*2),
                                   nn.ReLU(True))
        # 64 x 64
        self.layer3 = nn.Sequential(nn.Conv2d(ngf*2,ngf*4,kernel_size=3,stride=2,padding=1),
                                   InstanceNormalization(ngf*4),
                                   nn.ReLU(True))
        # 32 x 32
        self.layer4 = nn.Sequential(ResidualBlock(ngf*4),
                                    ResidualBlock(ngf*4),
                                    ResidualBlock(ngf*4),
                                    ResidualBlock(ngf*4),
                                    ResidualBlock(ngf*4),
                                    ResidualBlock(ngf*4))
        # 32 x 32
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, 1),
                                     InstanceNormalization(ngf*2),
                                     nn.ReLU(True))
        # 64 x 64
        self.layer6 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 1, 1),
                                    InstanceNormalization(ngf),
                                    nn.ReLU(True))
        # 128 x 128
        self.layer7 = nn.Sequential(nn.ReflectionPad2d((3,3,3,3)),
                                     nn.Conv2d(ngf,output_nc,kernel_size=7,stride=1),
                                     nn.Tanh())
        # 128 x 128
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        return out
