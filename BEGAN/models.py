import torch.nn as nn

def conv_block(in_dim,out_dim):
    return nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.MaxPool2d(2,2))
def deconv_block(in_dim,out_dim):
    return nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.UpsamplingNearest2d(scale_factor=2))

class Discriminator(nn.Module):
    def __init__(self,nc,ndf):
        super(Discriminator,self).__init__()
        # 64 x 64
        self.conv1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1),
                                    nn.ELU(True),
                                    conv_block(ndf,ndf))
        # 32 x 32
        self.conv2 = conv_block(ndf, ndf*2)
        # 16 x 16
        self.conv3 = conv_block(ndf*2, ndf*3)
        # 8 x 8
        self.conv4 = conv_block(ndf*3, ndf*4)

        self.embed1 = nn.Linear(ndf*4*8*8, 128)
        self.embed2 = nn.Linear(128, ndf*8*8)

        # 8 x 8
        self.deconv1 = deconv_block(ndf, ndf)
        # 16 x 16
        self.deconv2 = deconv_block(ndf, ndf)
        # 32 x 32
        self.deconv3 = deconv_block(ndf, ndf)
        # 64 x 64
        self.deconv4 = nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ndf, nc, kernel_size=3, stride=1, padding=1))


    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), ndf*4 * 8 * 8)
        out = self.embed1(out)
        out = self.embed2(out)
        out = out.view(out.size(0), ndf, 8, 8)
        out = self.deconv1(x)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        return out

class Generator(nn.Module):
    def __init__(self,nc,ndf):
        super(Generator,self).__init__()
        self.embed1 = nn.Linear(128, ndf*8*8)

        # 8 x 8
        self.deconv1 = deconv_block(ndf, ndf)
        # 16 x 16
        self.deconv2 = deconv_block(ndf, ndf)
        # 32 x 32
        self.deconv3 = deconv_block(ndf, ndf)
        # 64 x 64
        self.deconv4 = nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ndf, nc, kernel_size=3, stride=1, padding=1))


    def forward(self,x):
        out = self.embed1(out)
        out = out.view(out.size(0), ndf, 8, 8)
        out = self.deconv1(x)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        return out
