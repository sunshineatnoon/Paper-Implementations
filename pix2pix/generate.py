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
from model.Generator import Generator
from utils.dataset import Facades

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=6, help='input batch size')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--loadSize', type=int, default=286, help='scale image to this size')
parser.add_argument('--fineSize', type=int, default=256, help='random crop image to this size')
parser.add_argument('--input_nc', type=int, default=3, help='channel number of input image')
parser.add_argument('--output_nc', type=int, default=3, help='channel number of output image')
parser.add_argument('--flip', type=int, default=0, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--dataPath', default='facades/train/', help='path to training images')
parser.add_argument('--which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--outf', default='samples/', help='folder to output images and model checkpoints')
parser.add_argument('--ngf', type=int, default=64)

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

###########   Load netG   ###########
assert opt.netG != '', "netG must be provided!"
netG = Generator(opt.input_nc, opt.output_nc, opt.ngf)
netG.load_state_dict(torch.load(opt.netG))
###########   Generate   ###########
facades = Facades(opt.dataPath,opt.loadSize,opt.fineSize,opt.flip)
train_loader = torch.utils.data.DataLoader(dataset=facades,
                                           batch_size=opt.batchSize,
                                           shuffle=True,
                                           num_workers=2)
if(opt.cuda):
    netG.cuda()

for i, image in enumerate(train_loader):
    if(opt.which_direction == 'AtoB'):
        imgA = image[1]
        imgB = image[0]
    else:
        imgA = image[0]
        imgB = image[1]
    imgA = Variable(imgA)
    if(opt.cuda):
        imgA = imgA.cuda()
    fake = netG(imgA)

    vutils.save_image(fake.data,
                '%s/fakeB.png' % (opt.outf),
                normalize=True)
    vutils.save_image(imgA.data,
                '%s/A.png' % (opt.outf),
                normalize=True)
    vutils.save_image(imgB,
                '%s/realB.png' % (opt.outf),
                normalize=True)
    break
