#Credit: code copied from https://github.com/mrzhu-cool/pix2pix-pytorch/blob/master/models.py
import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Generator,self).__init__()
        '''
        # 256 x 256
        self.e1 = nn.Sequential(nn.Conv2d(input_nc,ngf,kernel_size=4,stride=2,padding=1),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 128 x 128
        self.e2 = nn.Sequential(nn.Conv2d(ngf,ngf*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf*2),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 64 x 64
        self.e3 = nn.Sequential(nn.Conv2d(ngf*2,ngf*4,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf*4),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 32 x 32
        self.e4 = nn.Sequential(nn.Conv2d(ngf*4,ngf*8,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf*8),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 16 x 16
        self.e5 = nn.Sequential(nn.Conv2d(ngf*8,ngf*8,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf*8),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 8 x 8
        self.e6 = nn.Sequential(nn.Conv2d(ngf*8,ngf*8,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf*8),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 4 x 4
        self.e7 = nn.Sequential(nn.Conv2d(ngf*8,ngf*8,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf*8),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 2 x 2
        self.e8 = nn.Sequential(nn.Conv2d(ngf*8,ngf*8,kernel_size=4,stride=2,padding=1),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 1 x 1
        self.d1 = nn.Sequential(nn.ConvTranspose2d(ngf*8,ngf*8,kernel_size=4,stride=2,padding=1),
                                nn.BatchNorm2d(ngf*8),
                                nn.Dropout())
        # 2 x 2
        self.d2 = nn.Sequential(nn.ConvTranspose2d(ngf*8*2,ngf*8,kernel_size=4,stride=2,padding=1),
                                nn.BatchNorm2d(ngf*8),
                                nn.Dropout())
        # 4 x 4
        self.d3 = nn.Sequential(nn.ConvTranspose2d(ngf*8*2,ngf*8,kernel_size=4,stride=2,padding=1),
                                nn.BatchNorm2d(ngf*8),
                                nn.Dropout())
        # 8 x 8
        self.d4 = nn.Sequential(nn.ConvTranspose2d(ngf*8*2,ngf*8,kernel_size=4,stride=2,padding=1),
                                nn.BatchNorm2d(ngf*8))
        # 16 x 16
        self.d5 = nn.Sequential(nn.ConvTranspose2d(ngf*8*2,ngf*4,kernel_size=4,stride=2,padding=1),
                                nn.BatchNorm2d(ngf*4))
        # 32 x 32
        self.d6 = nn.Sequential(nn.ConvTranspose2d(ngf*4*2,ngf*2,kernel_size=4,stride=2,padding=1),
                                nn.BatchNorm2d(ngf*2))
        # 64 x 64
        self.d7 = nn.Sequential(nn.ConvTranspose2d(ngf*2*2,ngf,kernel_size=4,stride=2,padding=1),
                                nn.BatchNorm2d(ngf))
        # 128 x 128
        self.d8 = nn.ConvTranspose2d(ngf*2,output_nc,kernel_size=4,stride=2,padding=1)
        # 256 x 256

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self,x):
        # encoder
        out_e1 = self.e1(x)              # 128 x 128
        out_e2 = self.e2(out_e1)             # 64 x 64
        out_e3 = self.e3(out_e2)             # 32 x 32
        out_e4 = self.e4(out_e3)             # 16 x 16
        out_e5 = self.e5(out_e4)             # 8 x 8
        out_e6 = self.e6(out_e5)             # 4 x 4
        out_e7 = self.e7(out_e6)             # 2 x 2
        out_e8 = self.e8(out_e7)             # 1 x 1

        # decoder
        out_d1 = self.d1(self.relu(out_e8))  # 2 x 2
        out_d1_ = torch.cat((out_d1, out_e7),1)

        out_d2 = self.d2(self.relu(out_d1_)) # 4 x 4
        out_d2_ = torch.cat((out_d2, out_e6),1)

        out_d3 = self.d3(self.relu(out_d2_)) # 8 x 8
        out_d3_ = torch.cat((out_d3, out_e5),1)

        out_d4 = self.d4(self.relu(out_d3_)) # 16 x 16
        out_d4_ = torch.cat((out_d4, out_e4),1)

        out_d5 = self.d5(self.relu(out_d4_)) # 32 x 32
        out_d5_ = torch.cat((out_d5, out_e3),1)

        out_d6 = self.d6(self.relu(out_d5_)) # 64 x 64
        out_d6_ = torch.cat((out_d6, out_e2),1)

        out_d7 = self.d7(self.relu(out_d6_)) # 128 x 128
        out_d7_ = torch.cat((out_d7, out_e1),1)

        out_d8 = self.d8(self.relu(out_d7_)) # 256 x 256
        out = self.tanh(out_d8)

        return out
        '''
        self.conv1 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv7 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv8 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1)
        self.dconv6 = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1)
        self.dconv7 = nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1)
        self.dconv8 = nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1)

        self.batch_norm = nn.BatchNorm2d(ngf)
        self.batch_norm2 = nn.BatchNorm2d(ngf * 2)
        self.batch_norm4 = nn.BatchNorm2d(ngf * 4)
        self.batch_norm8 = nn.BatchNorm2d(ngf * 8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        e1 = self.conv1(input)
        # state size is (ngf) x 128 x 128
        e2 = self.batch_norm2(self.conv2(self.leaky_relu(e1)))
        # state size is (ngf x 2) x 64 x 64
        e3 = self.batch_norm4(self.conv3(self.leaky_relu(e2)))
        # state size is (ngf x 4) x 32 x 32
        e4 = self.batch_norm8(self.conv4(self.leaky_relu(e3)))
        # state size is (ngf x 8) x 16 x 16
        e5 = self.batch_norm8(self.conv5(self.leaky_relu(e4)))
        # state size is (ngf x 8) x 8 x 8
        e6 = self.batch_norm8(self.conv6(self.leaky_relu(e5)))
        # state size is (ngf x 8) x 4 x 4
        e7 = self.batch_norm8(self.conv7(self.leaky_relu(e6)))
        # state size is (ngf x 8) x 2 x 2
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 1 x 1
        d1_ = self.dropout(self.batch_norm8(self.dconv1(self.relu(e8))))
        # state size is (ngf x 8) x 2 x 2
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.dropout(self.batch_norm8(self.dconv2(self.relu(d1))))
        # state size is (ngf x 8) x 4 x 4
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.dropout(self.batch_norm8(self.dconv3(self.relu(d2))))
        # state size is (ngf x 8) x 8 x 8
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.batch_norm8(self.dconv4(self.relu(d3)))
        # state size is (ngf x 8) x 16 x 16
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.batch_norm4(self.dconv5(self.relu(d4)))
        # state size is (ngf x 4) x 32 x 32
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.batch_norm2(self.dconv6(self.relu(d5)))
        # state size is (ngf x 2) x 64 x 64
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.batch_norm(self.dconv7(self.relu(d6)))
        # state size is (ngf) x 128 x 128
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.relu(d7))
        # state size is (nc) x 256 x 256
        output = self.tanh(d8)
        return output
