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
from models import Discriminator
from models import Generator
from data.dataset import CelebA

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--loadSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--fineSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--flip', type=int, default=1, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='dcgan/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dataPath', default='data/CelebA/images/', help='which dataset to train on')
parser.add_argument('--lambda_k', type=float, default=0.001, help='learning rate of k')
parser.add_argument('--gamma', type=float, default=0.5, help='balance bewteen D and G')

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
dataset = CelebA(opt.dataPath,opt.loadSize,opt.fineSize,opt.flip)
loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=opt.batchSize,
                                           shuffle=True,
                                           num_workers=2)


###############   MODEL   ####################
ndf = opt.ndf
ngf = opt.ngf
nc = 3

netD = Discriminator(nc, ndf)
netG = Generator(nc, ngf, opt.nz)
if(opt.cuda):
    netD.cuda()
    netG.cuda()

###########   LOSS & OPTIMIZER   ##########
criterion = nn.L1Loss()
optimizerD = torch.optim.Adam(netD.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))

##########   GLOBAL VARIABLES   ###########
noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
real = torch.FloatTensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

noise = Variable(noise)
real = Variable(real)
label = Variable(label)
if(opt.cuda):
    noise = noise.cuda()
    real = real.cuda()
    label = label.cuda()

########### Training   ###########
k = 0
for epoch in range(1,opt.niter+1):
    for i, images in enumerate(loader):
        netD.zero_grad()
        netG.zero_grad()

        real.data.resize_(images.size()).copy_(images)

        # generate fake data
        noise.data.resize_(images.size(0), opt.nz, 1, 1)
        noise.data.normal_(0,1)
        fake = netG(noise)

        fake_recons = netD(fake)
        real_recons = netD(fake)

        errD_real = torch.mean(criterion(torch.abs(real_recons-real)))
        errD_fake = torch.mean(criterion(torch.abs(fake_recons-fake)))

        errD = errD_real - k*errD_fake
        errG = errD_fake

        balance = (opt.gamma * errD_real - errD_fake).data[0]
        k = np.clip(k + opt.lambda_k * balance,0,1)

        measure = errD_real.data[0] + torch.abs(balance)

        errD.backward()
        optimizerD.step()

        errG.backward()
        optimizerG.step()

        ########### Logging #########
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Measure: %.4f '
                  % (epoch, opt.niter, i, len(loader),
                     errD.data[0], errG.data[0], measure))

        ########## Visualize #########
        if(i % 50 == 0):
            vutils.save_image(fake.data,
                        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)

torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
