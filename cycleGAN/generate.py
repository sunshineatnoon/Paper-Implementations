from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from utils.dataset import DATASET 
from model.Discriminator import Discriminator
from model.Generator import Generator

parser = argparse.ArgumentParser(description='train pix2pix model')
parser.add_argument('--batchSize', type=int, default=1, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='samples/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dataPath', default='facades/test/', help='path to training images')
parser.add_argument('--loadSize', type=int, default=128, help='scale image to this size')
parser.add_argument('--fineSize', type=int, default=128, help='random crop image to this size')
parser.add_argument('--flip', type=int, default=0, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--input_nc', type=int, default=3, help='channel number of input image')
parser.add_argument('--output_nc', type=int, default=3, help='channel number of output image')
parser.add_argument('--G_AB', default='', help='path to pre-trained G_AB')
parser.add_argument('--G_BA', default='', help='path to pre-trained G_BA')
parser.add_argument('--imgNum', type=int, default=32, help='image number')

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
##########   DATASET   ###########
datasetA = DATASET(os.path.join(opt.dataPath,'A'),opt.loadSize,opt.fineSize,opt.flip)
datasetB = DATASET(os.path.join(opt.dataPath,'B'),opt.loadSize,opt.fineSize,opt.flip)
loader_A = torch.utils.data.DataLoader(dataset=datasetA,
                                       batch_size=opt.batchSize,
                                       shuffle=True,
                                       num_workers=2)
loaderA = iter(loader_A)
loader_B = torch.utils.data.DataLoader(dataset=datasetB,
                                       batch_size=opt.batchSize,
                                       shuffle=True,
                                       num_workers=2)
loaderB = iter(loader_B)
###########   MODEL   ###########
ndf = opt.ndf
ngf = opt.ngf
nc = 3

G_AB = Generator(opt.input_nc, opt.output_nc, opt.ngf)
G_BA = Generator(opt.output_nc, opt.input_nc, opt.ngf)

if(opt.G_AB != ''):
    print('Warning! Loading pre-trained weights.')
    G_AB.load_state_dict(torch.load(opt.G_AB))
    G_BA.load_state_dict(torch.load(opt.G_BA))
else:
    print('ERROR! G_AB and G_BA must be provided!')

if(opt.cuda):
    G_AB.cuda()
    G_BA.cuda()


###########   GLOBAL VARIABLES   ###########
input_nc = opt.input_nc
output_nc = opt.output_nc
fineSize = opt.fineSize

real_A = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
real_B = torch.FloatTensor(opt.batchSize, output_nc, fineSize, fineSize)

real_A = Variable(real_A)
real_B = Variable(real_B)

if(opt.cuda):
    real_A = real_A.cuda()
    real_B = real_B.cuda()

###########   Testing    ###########
def test():
    AB_all = torch.Tensor(opt.imgNum, 3, opt.fineSize, opt.fineSize)
    ABA_all = torch.Tensor(opt.imgNum, 3, opt.fineSize, opt.fineSize)
    A_all = torch.Tensor(opt.imgNum, 3, opt.fineSize, opt.fineSize)
    if(opt.cuda):
	AB_all = AB_all.cuda()
	ABA_all = ABA_all.cuda()
	A_all = A_all.cuda()

    for i in range(0,opt.imgNum,opt.batchSize):
        imgA = loaderA.next() 
        imgB = loaderB.next()
        real_A.data.copy_(imgA)
        real_B.data.copy_(imgB)

        AB = G_AB(real_A)
        ABA = G_BA(AB)
        AB_all[i,:,:,:] = AB.data
        ABA_all[i,:,:,:] = ABA.data
        A_all[i,:,:,:] = imgA


    vutils.save_image(AB_all,
            '%s/AB.png' % (opt.outf),
            normalize=True,
	    nrow=4)
    vutils.save_image(ABA_all,
            '%s/ABA.png' % (opt.outf),
            normalize=True,
	    nrow=4)
    vutils.save_image(A_all,
            '%s/A.png' % (opt.outf),
            normalize=True,
	    nrow=4)

    BA_all = torch.Tensor(opt.imgNum, 3, opt.fineSize, opt.fineSize)
    BAB_all = torch.Tensor(opt.imgNum, 3, opt.fineSize, opt.fineSize)
    B_all = torch.Tensor(opt.imgNum, 3, opt.fineSize, opt.fineSize)
    if(opt.cuda):
	BA_all = AB_all.cuda()
	BAB_all = ABA_all.cuda()
	B_all = A_all.cuda()

    for i in range(0,opt.imgNum,opt.batchSize):
        imgA = loaderA.next() 
        imgB = loaderB.next()
        real_A.data.copy_(imgA)
        real_B.data.copy_(imgB)

        BA = G_BA(real_B)
        BAB = G_AB(BA)
        BA_all[i,:,:,:] = BA.data
        BAB_all[i,:,:,:] = BAB.data
        B_all[i,:,:,:] = imgB


    vutils.save_image(BA_all,
            '%s/BA.png' % (opt.outf),
            normalize=True,
	    nrow=4)
    vutils.save_image(BAB_all,
            '%s/BAB.png' % (opt.outf),
            normalize=True,
	    nrow=4)
    vutils.save_image(B_all,
            '%s/B.png' % (opt.outf),
            normalize=True,
	    nrow=4)

test()

