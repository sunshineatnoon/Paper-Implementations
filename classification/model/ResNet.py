import torch
import torch.nn as nn

def conv3x3(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)

class ResidualBlock(nn.Module):
    '''
    A single Residual Block:
    in ->
        conv(in_channels,out_channels,stride) -> BN -> ReLU
        conv(out_channels,out_channels,1) -> BN
    -> out
    (downsample)in + out
    '''
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock,self).__init__()
        self.conv1 = conv3x3(in_channels,out_channels,stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels,out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if(self.downsample):
            out = self.downsample(x) + out
        else:
            out = out + x

        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(ResidualBlock, 16, 1, layers[0])
        self.layer2 = self._make_layer(ResidualBlock, 32, 2, layers[1])
        self.layer3 = self._make_layer(ResidualBlock, 64, 2, layers[2])

        self.avgPool = nn.AvgPool2d(7)
        self.fc = nn.Linear(64, num_classes)


    def _make_layer(self, block, out_channels, stride, Nblock):
        '''
        Given a block, stack the block Nblock times to form a layer:
        1. Block(in_channels, out_channels, stride)
        2. Block(out_channels, out_channels, 1)
        ...
        Nblock. Block(out_channels, out_channels, 1)
        '''
        layers = []
        downsample = None
        if(self.in_channels != out_channels) or (stride != 1):
            downsample = nn.Sequential(
                        conv3x3(self.in_channels, out_channels, stride),
                        nn.BatchNorm2d(out_channels)
            )
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1,Nblock):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgPool(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out
