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
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--loadSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--fineSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--flip', type=int, default=1, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=300000, help='number of iterations to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='dcgan/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dataPath', default='data/CelebA/images/', help='which dataset to train on')
parser.add_argument('--lambda_k', type=float, default=0.001, help='learning rate of k')
parser.add_argument('--gamma', type=float, default=0.5, help='balance bewteen D and G')
parser.add_argument('--save_step', type=int, default=10000, help='save weights every 50000 iterations ')
parser.add_argument('--hidden_size', type=int, default=64, help='bottleneck dimension of Discriminator')
parser.add_argument('--lr_decay_every', type=int, default=3000, help='decay lr this many iterations')

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
loader_ = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=opt.batchSize,
                                           shuffle=True,
                                           num_workers=2)
loader = iter(loader_)

###############   MODEL   ####################
criterion = nn.L1Loss()
ndf = opt.ndf
ngf = opt.ngf
nc = 3

netD = Discriminator(nc, ndf, opt.hidden_size,opt.fineSize)
netG = Generator(nc, ngf, opt.nz,opt.fineSize)
if(opt.cuda):
    netD.cuda()
    netG.cuda()

###########   LOSS & OPTIMIZER   ##########
optimizerD = torch.optim.Adam(netD.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))

##########   GLOBAL VARIABLES   ###########
noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
real = torch.FloatTensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
label = torch.FloatTensor(1)

noise = Variable(noise)
real = Variable(real)
label = Variable(label)
if(opt.cuda):
    noise = noise.cuda()
    real = real.cuda()
    label = label.cuda()
    criterion.cuda()

########### Training   ###########
def adjust_learning_rate(optimizer, niter):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.lr * (0.95 ** (niter // opt.lr_decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

k = 0 
for iteration in range(1,opt.niter+1):
    try:
        images = loader.next()
    except StopIteration:
        loader = iter(loader_)
        images = loader.next()
    netD.zero_grad()

    real.data.resize_(images.size()).copy_(images)

    # generate fake data
    noise.data.resize_(images.size(0), opt.nz)
    noise.data.uniform_(-1,1)
    fake = netG(noise)

    fake_recons = netD(fake.detach())
    real_recons = netD(real)

    err_real = torch.mean(torch.abs(real_recons-real))
    err_fake = torch.mean(torch.abs(fake_recons-fake))

    errD = err_real - k*err_fake
    errD.backward()
    optimizerD.step()

    netG.zero_grad()
    fake = netG(noise)
    fake_recons = netD(fake)
    errG = torch.mean(torch.abs(fake_recons-fake))
    errG.backward()
    optimizerG.step()

    balance = (opt.gamma * err_real - err_fake).data[0]
    k = min(max(k + opt.lambda_k * balance,0),1)
    measure = err_real.data[0] + np.abs(balance)
    ########### Logging #########
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f Measure: %.4f K: %.4f LR: %.8f'
              % (iteration, opt.niter, 
                 errD.data[0], errG.data[0], measure, k, optimizerD.param_groups[0]['lr']))

    ########### Learning Rate Decay #########
    optimizerD = adjust_learning_rate(optimizerD,iteration)
    optimizerG = adjust_learning_rate(optimizerG,iteration)
    ########## Visualize #########
    if(iteration % 1000 == 0):
        vutils.save_image(fake.data,
                    '%s/fake_samples_iteration_%03d.png' % (opt.outf, iteration),
                    normalize=True)
        vutils.save_image(real.data,
                    '%s/real_samples_iteration_%03d.png' % (opt.outf, iteration),
                    normalize=True)
        vutils.save_image(real_recons.data,
                    '%s/real_recons_samples_iteration_%03d.png' % (opt.outf, iteration),
                    normalize=True)

    if(iteration % opt.save_step == 0):
        torch.save(netG.state_dict(), '%s/netG_%d.pth' % (opt.outf,iteration))
        torch.save(netD.state_dict(), '%s/netD_%d.pth' % (opt.outf,iteration))
