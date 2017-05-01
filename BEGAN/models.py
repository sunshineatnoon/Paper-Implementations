import torch.nn as nn

def conv_block(in_dim,out_dim):
    return nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0),
                         nn.AvgPool2d(kernel_size=2,stride=2))
def deconv_block(in_dim,out_dim):
    return nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                         nn.ELU(True),
                         nn.UpsamplingNearest2d(scale_factor=2))

class Discriminator(nn.Module):
    def __init__(self,nc,ndf,hidden_size,imageSize):
        super(Discriminator,self).__init__()
        # 64 x 64 
        self.conv1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1),
                                    nn.ELU(True),
                                    conv_block(ndf,ndf))
        # 32 x 32 
        self.conv2 = conv_block(ndf, ndf*2)
        # 16 x 16 
        self.conv3 = conv_block(ndf*2, ndf*3)
        if(imageSize == 64):
            # 8 x 8
            self.conv4 = nn.Sequential(nn.Conv2d(ndf*3,ndf*3,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ndf*3,ndf*3,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True)) 
            self.embed1 = nn.Linear(ndf*3*8*8, hidden_size)
        else:
            self.conv4 = conv_block(ndf*3, ndf*4)
            self.conv5 = nn.Sequential(nn.Conv2d(ndf*4,ndf*4,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ndf*4,ndf*4,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True)) 
            self.embed1 = nn.Linear(ndf*4*8*8, hidden_size)
        self.embed2 = nn.Linear(hidden_size, ndf*8*8)

        # 8 x 8
        self.deconv1 = deconv_block(ndf, ndf)
        # 16 x 16
        self.deconv2 = deconv_block(ndf, ndf)
        # 32 x 32
        self.deconv3 = deconv_block(ndf, ndf)
        if(imageSize == 64):
        # 64 x 64
            self.deconv4 = nn.Sequential(nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ndf, nc, kernel_size=3, stride=1, padding=1))
        else:
            self.deconv4 = deconv_block(ndf, ndf)
            self.deconv5 = nn.Sequential(nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ndf, nc, kernel_size=3, stride=1, padding=1),
                             nn.Tanh())

	self.ndf = ndf
        self.imageSize = imageSize

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        if(self.imageSize == 128):
            out = self.conv5(out)
            out = out.view(out.size(0), self.ndf*4 * 8 * 8)
        else:
            out = out.view(out.size(0), self.ndf*3 * 8 * 8)
        out = self.embed1(out)
        out = self.embed2(out)
        out = out.view(out.size(0), self.ndf, 8, 8)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        if(self.imageSize == 128):
            out = self.deconv5(out)
        return out

class Generator(nn.Module):
    def __init__(self,nc,ngf,nz,imageSize):
        super(Generator,self).__init__()
        self.embed1 = nn.Linear(nz, ngf*8*8)

        # 8 x 8
        self.deconv1 = deconv_block(ngf, ngf)
        # 16 x 16
        self.deconv2 = deconv_block(ngf, ngf)
        # 32 x 32
        self.deconv3 = deconv_block(ngf, ngf)
        if(imageSize == 128):
            self.deconv4 = deconv_block(ngf, ngf)
            # 128 x 128 
            self.deconv5 = nn.Sequential(nn.Conv2d(ngf,ngf,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ngf,ngf,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1))
        else:
            self.deconv4 = nn.Sequential(nn.Conv2d(ngf,ngf,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ngf,ngf,kernel_size=3,stride=1,padding=1),
                             nn.ELU(True),
                             nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1),
                             nn.Tanh())
        self.ngf = ngf
        self.imageSize = imageSize

    def forward(self,x):
        out = self.embed1(x)
        out = out.view(out.size(0), self.ngf, 8, 8)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        if(self.imageSize == 128):
            out = self.deconv5(out)
        return out
