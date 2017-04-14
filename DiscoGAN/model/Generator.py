import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Generator,self).__init__()
        # 64 x 64
        self.conv1 = nn.Sequential(nn.Conv2d(input_nc,ngf,kernel_size=4,stride=2,padding=1,bias=False),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 32 x 32
        self.conv2 = nn.Sequential(nn.Conv2d(ngf,ngf*2,kernel_size=4,stride=2,padding=1,bias=False),
                                 nn.BatchNorm2d(ngf*2),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 16 x 16
        self.conv3 = nn.Sequential(nn.Conv2d(ngf*2,ngf*4,kernel_size=4,stride=2,padding=1,bias=False),
                                 nn.BatchNorm2d(ngf*4),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 8 x 8
        self.conv4 = nn.Sequential(nn.Conv2d(ngf*4,ngf*8,kernel_size=4,stride=2,padding=1,bias=False),
                                 nn.BatchNorm2d(ngf*8),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 4 x 4
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(ngf*4),
                                     nn.ReLU(True))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(ngf*2),
                                     nn.ReLU(True))
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(ngf),
                                     nn.ReLU(True))
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(ngf, output_nc, 4, 2, 1, bias=False),
                                     nn.Tanh())

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        return out

