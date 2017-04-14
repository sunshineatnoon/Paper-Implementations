import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,input_nc,ndf):
        super(Discriminator,self).__init__()
        # 64 x 64
        self.layer1 = nn.Sequential(nn.Conv2d(input_nc,ndf,kernel_size=4,stride=2,padding=1,bias=False),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 32 x 32
        self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1,bias=False),
                                 nn.BatchNorm2d(ndf*2),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 16 x 16
        self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1,bias=False),
                                 nn.BatchNorm2d(ndf*4),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 8 x 8
        self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=2,padding=1,bias=False),
                                 nn.BatchNorm2d(ndf*8),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 4 x 4
        self.layer5 = nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=4,stride=1,padding=0,bias=False),
                                 nn.Sigmoid())
        # 1 x 1

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

