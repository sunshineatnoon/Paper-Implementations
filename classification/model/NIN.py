import torch.nn as nn
import torch

class NINBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(NINBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class NIN(nn.Module):
    def __init__(self):
        super(NIN,self).__init__()
        self.block1 = nn.Sequential(NINBlock(1, 192, 5, 1, 2),
                                    NINBlock(192, 160, 1, 1, 0),
                                    NINBlock(160, 96, 1, 1, 0),
                                    nn.MaxPool2d(3,2,ceil_mode=True),
                                    nn.Dropout(inplace = True))

        self.block2 = nn.Sequential(NINBlock(96, 192, 5, 1, 2),
                                    NINBlock(192, 192, 1, 1, 0),
                                    NINBlock(192, 192, 1, 1, 0),
                                    nn.AvgPool2d(3,2,ceil_mode=True),
                                    nn.Dropout(inplace = True))

        self.block3 = nn.Sequential(NINBlock(192, 192, 3, 1, 1),
                                    NINBlock(192, 192, 1, 1, 0),
                                    NINBlock(192, 10, 1, 1, 0),
                                    nn.AvgPool2d(7,1,ceil_mode=True))

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out.view(out.size(0),10)
        out = torch.squeeze(out)
        return out
