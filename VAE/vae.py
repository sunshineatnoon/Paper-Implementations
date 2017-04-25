from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='samples/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--log_interval', type=int, default=50, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

###############   DATASET   ##################
dataset = dset.MNIST(root = '../data/',
                     transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]),
                      download = True)


loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = opt.batchSize,
                                     shuffle = True)

###############   MODEL   ##################
class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        # 28 x 28
        n = 64
        self.conv1 = nn.Sequential(nn.Conv2d(1,n,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(n),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 14 x 14
        self.conv2 = nn.Sequential(nn.Conv2d(n,n*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(n*2),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 7 x 7
        self.conv3 = nn.Sequential(nn.Conv2d(n*2,n,kernel_size=3,stride=1,padding=1),
                                 nn.BatchNorm2d(n),
                                 nn.LeakyReLU(0.2,inplace=True))

        self.fc11 = nn.Linear(n * 7 * 7, 20)
        self.fc12 = nn.Linear(n * 7 * 7, 20)
        self.fc2 = nn.Linear(20, n * 7 * 7)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(n,n,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(n),
                                 nn.ReLU())
        # 14 x 14
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(n,1,kernel_size=4,stride=2,padding=1),
                                 nn.Sigmoid())
        # 28 x 28

    def encoder(self, x):
        # input: noise output: mu and sigma
        # opt.batchSize x 1 x 28 x 28
        out = self.conv1(x)
        # opt.batchSize x n x 14 x 14
        out = self.conv2(out)
        # opt.batchSize x n x 7 x 7
        out = self.conv3(out)
        return self.fc11(out.view(out.size(0),-1)),self.fc12(out.view(out.size(0),-1))

    def sampler(self, mu, logvar):
        var = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_()
        eps = Variable(eps)
        if(opt.cuda):
            eps = eps.cuda()
        return eps.mul(var).add_(mu)

    def decoder(self, x):
        out = self.fc2(x)
        out = self.deconv1(out.view(x.size(0), 64, 7, 7))
        # opt.batchSize x n x 7 x 7
        out = self.deconv2(out)
        # opt.batchSize x n x 14 x 14
        return out
        # opt.batchSize x 1 x 28 x 28

    def forward(self, x):
        mu, logvar = self.encoder(x)
        out = self.sampler(mu, logvar)
        out = self.decoder(out)
        return out, mu, logvar

model = VAE()
if(opt.cuda):
    mode.cuda()
###########   LOSS & OPTIMIZER   ##########
bce = nn.BCELoss()
bce.size_average = False
if(opt.cuda):
    bce.cuda()
def LossFunction(out, target, mu, logvar):
    bceloss = bce(out, target)
    kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    kldloss = torch.sum(kld).mul_(-0.5)
    return bceloss + kldloss
optimizer = optim.Adam(model.parameters(),lr=opt.lr)

##########   GLOBAL VARIABLES   ###########
data = torch.Tensor(opt.batchSize, opt.imageSize * opt.imageSize)
data = Variable(data)
if(opt.cuda):
    data = data.cuda()
###############   TRAINING   ##################
def sample(epoch):
    model.eval()
    eps = torch.FloatTensor(opt.batchSize, 20).normal_()
    eps = Variable(eps)
    if(opt.cuda):
        eps = eps.cuda()
    fake = model.decoder(eps)
    vutils.save_image(fake.data.resize_(opt.batchSize,1,opt.imageSize,opt.imageSize),
                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                normalize=True)

def train(epoch):
    model.train()
    for i, (images,_) in enumerate(loader):
        model.zero_grad()
        data.data.resize_(images.size()).copy_(images)
        recon, mu, logvar = model(data)
        loss = LossFunction(recon, data, mu, logvar)
        loss.backward()
        optimizer.step()
        if i % opt.log_interval == 0:
            sample(epoch)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(loader.dataset),
                100. * i / len(loader),
                loss.data[0] / len(data)))

for epoch in range(1, opt.niter + 1):
    train(epoch)
