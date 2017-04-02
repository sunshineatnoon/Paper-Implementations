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

from utils.dataset import Facades
from model.Discriminator import Discriminator
from model.Generator import Generator


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='checkpoints/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dataPath', default='facades/train/', help='path to training images')
parser.add_argument('--loadSize', type=int, default=286, help='scale image to this size')
parser.add_argument('--fineSize', type=int, default=256, help='random crop image to this size')
parser.add_argument('--flip', type=int, default=1, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--input_nc', type=int, default=3, help='channel number of input image')
parser.add_argument('--output_nc', type=int, default=3, help='channel number of output image')
parser.add_argument('--which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--lamb', default=100, help='weight on L1 term in objective')

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

###########   DATASET   ###########
facades = Facades(opt.dataPath,opt.loadSize,opt.fineSize,opt.flip)
train_loader = torch.utils.data.DataLoader(dataset=facades,
                                           batch_size=opt.batchSize,
                                           shuffle=True,
                                           num_workers=2)

###########   MODEL   ###########
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

ndf = opt.ndf
ngf = opt.ngf
nc = 3

netD = Discriminator(opt.input_nc,opt.output_nc,ndf)
netG = Generator(opt.input_nc, opt.output_nc, opt.ngf)
if(opt.cuda):
    netD.cuda()
    netG.cuda()

netG.apply(weights_init)
netD.apply(weights_init)
print(netD)
print(netG)

###########   LOSS & OPTIMIZER   ##########
criterion = nn.BCELoss()
criterionL1 = nn.L1Loss()
optimizerD = torch.optim.Adam(netD.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))

###########   GLOBAL VARIABLES   ###########
input_nc = opt.input_nc
output_nc = opt.output_nc
fineSize = opt.fineSize

real_A = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
real_B = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
label = torch.FloatTensor(opt.batchSize)

real_A = Variable(real_A)
real_B = Variable(real_B)
label = Variable(label)

if(opt.cuda):
    real_A = real_A.cuda()
    real_B = real_B.cuda()
    label = label.cuda()

real_label = 1
fake_label = 0

########### Training   ###########
netD.train()
netG.train()
for epoch in range(1,opt.niter+1):
    for i, image in enumerate(train_loader):
        ########### fDx ###########
        netD.zero_grad()
        if(opt.which_direction == 'AtoB'):
            imgA = image[1]
            imgB = image[0]
        else:
            imgA = image[0]
            imgB = image[1]

        # train with real data
        real_A.data.resize_(imgA.size()).copy_(imgA)
        real_B.data.resize_(imgB.size()).copy_(imgB)
        real_AB = torch.cat((real_A, real_B), 1)


        output = netD(real_AB)
        label.data.resize_(output.size())
        label.data.fill_(real_label)
        errD_real = criterion(output, label)
        errD_real.backward()

        # train with fake
        fake_B = netG(real_A)
        fake_AB = torch.cat((real_A, fake_B), 1)
        output = netD(fake_AB.detach())
        errD_fake = criterion(output,label)
        errD_fake.backward()

        errD = errD_fake + errD_real
        optimizerD.step()

        ########### fGx ###########
        netG.zero_grad()
        label.data.fill_(real_label)
        output = netD(fake_AB)
        errGAN = criterion(output, label)
        errL1 = criterionL1(fake_B,real_B)
        errG = errGAN + opt.lamb*errL1

        errG.backward()
        optimizerG.step()

        ########### Logging ##########
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_L1: %.4f'
                  % (epoch, opt.niter, i, len(train_loader),
                     errD.data[0], errG.data[0], errL1.data[0]))

    ########## Visualize #########
    if(epoch % 5 == 0):
        vutils.save_image(fake_B.data,
                    'fake_samples_epoch_%03d.png' % (epoch),
                    normalize=True)

torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
