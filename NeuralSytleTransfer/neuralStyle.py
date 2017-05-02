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
from PIL import Image
import numpy as np
import torchvision.models as models
from vgg import VGG


parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', type=int, default=512, help='the height / width of the input image to network')
parser.add_argument('--style_image', default="images/picasso.jpg", help='path to style image')
parser.add_argument('--content_image', default="images/dancing.jpg", help='path to style image')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e1, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='images/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--content_weight', type=int, default=5e0, help='content loss weight')
parser.add_argument('--style_weight', type=int, default=1e2, help='style loss weight')
parser.add_argument('--content_layers', default='r42', help='layers for content')
parser.add_argument('--style_layers', default='r11,r21,r31,r41,r51', help='layers for style')
parser.add_argument('--vgg_dir', default='models/vgg_conv.pth', help='path to pretrained VGG19 net')

opt = parser.parse_args()
# turn content layers and style layers to a list
opt.content_layers = opt.content_layers.split(',')
opt.style_layers = opt.style_layers.split(',')
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
transform = transforms.Compose([
    transforms.Scale(opt.imageSize),
    transforms.ToTensor()
    ])
def load_image(path):
    img = Image.open(path)
    img = Variable(transform(img))
    img = img.unsqueeze(0)
    return img
styleImg = load_image(opt.style_image) # 1x3x512x512
contentImg = load_image(opt.content_image) # 1x3x512x512
if(opt.cuda):
    styleImg = styleImg.cuda()
    contentImg = contentImg.cuda()

###############   MODEL   ####################
vgg = VGG()
vgg.load_state_dict(torch.load(opt.vgg_dir))
for param in vgg.parameters():
    param.requires_grad = False
if(opt.cuda):
    vgg.cuda()
###########   LOSS & OPTIMIZER   ##########
class GramMatrix(nn.Module):
    def forward(self,input):
        b, c, h, w = input.size()
        f = input.view(b,c,h*w) # bxcx(hxw)
        # torch.bmm(batch1, batch2, out=None)   #
        # batch1: bxmxp, batch2: bxpxn -> bxmxn #
        G = torch.bmm(f,f.transpose(1,2)) # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
        return G.div_(h*w)

class styleLoss(nn.Module):
    def forward(self,input,target):
        GramInput = GramMatrix()(input)
        return nn.MSELoss()(GramInput,target)

styleTargets = []
for t in vgg(styleImg,opt.style_layers):
    t = t.detach()
    styleTargets.append(GramMatrix()(t))
contentTargets = []
for t in vgg(contentImg,opt.content_layers):
    t = t.detach()
    contentTargets.append(t)

styleLosses = [styleLoss()] * len(opt.style_layers)
contentLosses = [nn.MSELoss()] * len(opt.content_layers)

# summary style and content loss so that we only need to go through the vgg once to get
# all style losses and content losses
losses = styleLosses + contentLosses
targets = styleTargets + contentTargets
loss_layers = opt.style_layers + opt.content_layers
weights = [opt.style_weight]*len(opt.style_layers) + [opt.content_weight]*len(opt.content_layers)

optImg = Variable(contentImg.data.clone(), requires_grad=True)
optimizer = optim.LBFGS([optImg]);

# shift everything to cuda if possible
if(opt.cuda):
    for loss in losses:
        loss = loss.cuda()
    optImg.cuda()
###########   TRAINING   ##########

for iteration in range(1,opt.niter+1):
    print('Iteration [%d]/[%d]'%(iteration,opt.niter))
    def closure():
        optimizer.zero_grad()
        out = vgg(optImg,loss_layers)
        totalLossList = []
        for i in range(len(out)):
            layer_output = out[i]
            loss_i = losses[i]
            target_i = targets[i]
            totalLossList.append(loss_i(layer_output,target_i) * weights[i])
        totalLoss = sum(totalLossList)
        totalLoss.backward()
        print('loss: %f'%(totalLoss.data[0]))
        return totalLoss
    optimizer.step(closure)

vutils.save_image(optImg.data,
            '%s/transferred.png' % (opt.outf),
            normalize=True)
