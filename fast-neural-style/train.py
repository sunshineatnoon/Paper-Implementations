from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torchvision.models as models

from utils.vgg import VGG
from utils.transformer import TransformerNet

parser = argparse.ArgumentParser()
parser.add_argument("--style_image", default="images/picasso.jpg", help='path to style image')
parser.add_argument("--outf", default="images/", help='folder to output images and model checkpoints')
parser.add_argument("--dataPath", default="data/", help='folder to training image')
parser.add_argument("--content_layers", default="r33", help='layers for content')
parser.add_argument("--style_layers", default="r12,r22,r33,r43", help='layers for style')
parser.add_argument("--batchSize", type=int,default=4, help='batch size')
parser.add_argument("--niter", type=int,default=40000, help='iterations to train the model')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--imageSize', type=int, default=256, help='image size')
parser.add_argument("--vgg_dir", default="models/vgg_conv.pth", help='path to pretrained VGG19 net')
parser.add_argument("--lr", type=float, default=1e-3, help='learning rate, default=0.0002')
parser.add_argument("--content_weight", type=float, default=1.0, help='content loss weight')
parser.add_argument("--style_weight", type=float, default=5.0, help='style loss weight')
parser.add_argument("--save_image_every", type=int, default=5, help='save transferred image every this much times')
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
    transforms.CenterCrop(opt.imageSize),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul_(255)),
    ])

train_dataset = datasets.ImageFolder(opt.dataPath, transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                            batch_size=opt.batchSize,
                            shuffle=True,
                            num_workers=2)
loader = iter(train_loader)

# style image
def load_image(path,style=False):
    img = Image.open(path)
    img = Variable(transform(img))
    img = img.unsqueeze(0)
    return img

styleImg = load_image(opt.style_image) # 1x3x512x512
if(opt.cuda):
    styleImg = styleImg.cuda()

###########   MODEL   ###########
## pre-trained VGG net
vgg = VGG()
vgg.load_state_dict(torch.load(opt.vgg_dir))
for param in vgg.parameters():
    param.requires_grad = False

## transformer net
cnn = TransformerNet()

if(opt.cuda):
    vgg.cuda()
    cnn.cuda()

###########   LOSS & OPTIMIZER   ##########
class GramMatrix(nn.Module):
    def forward(self,input):
        b, c, h, w = input.size()
        f = input.view(b,c,h*w) # bxcx(hxw)
        # torch.bmm(batch1, batch2, out=None)   #
        # batch1: bxmxp, batch2: bxpxn -> bxmxn #
        G = torch.bmm(f,f.transpose(1,2)) # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
        return G.div_(c*h*w)

class styleLoss(nn.Module):
    def forward(self,input,target):
        GramInput = GramMatrix()(input)
        return nn.MSELoss()(GramInput,target)

styleTargets = []
for t in vgg(styleImg,opt.style_layers):
    t = t.detach()
    temp = GramMatrix()(t)
    temp = temp.repeat(opt.batchSize,1,1,1)
    styleTargets.append(temp)
styleLosses = [styleLoss()] * len(opt.style_layers)
contentLosses = [nn.MSELoss()] * len(opt.content_layers)
losses = styleLosses + contentLosses

loss_layers = opt.style_layers + opt.content_layers
weights = [opt.style_weight]*len(opt.style_layers) + [opt.content_weight]*len(opt.content_layers)

optimizer = optim.Adam(cnn.parameters(), opt.lr)

# shift everything to cuda if possible
if(opt.cuda):
    for loss in losses:
        loss = loss.cuda()

###########   GLOBAL VARIABLES   ###########
images = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
images = Variable(images)
if(opt.cuda):
    images = images.cuda()

###########   Training   ###########
for iteration in range(1,opt.niter+1):
    optimizer.zero_grad()
    try:
        img,_ =  loader.next()
    except StopIteration:
        loader = iter(train_loader)
        img,_ = loader.next()
    if(img.size(0) < opt.batchSize):
        continue

    images.data.resize_(img.size()).copy_(img)

    optImg = cnn(images)

    contentTargets = []
    for t in vgg(images,opt.content_layers):
        t = t.detach()
        contentTargets.append(t)
    targets = styleTargets + contentTargets

    out = vgg(optImg, loss_layers)
    totalLoss = 0
    for i in range(len(out)):
        layer_output = out[i]
        loss_i = losses[i]
        target_i = targets[i]
        totalLoss += loss_i(layer_output,target_i) * weights[i]
    totalLoss.backward()
    print('loss: %f'%(totalLoss.data[0]))
    optimizer.step()

    # save transffered image
    if(iteration % opt.save_image_every == 0):
        optImg = optImg.clamp(0,255)
        vutils.save_image(optImg.data,
            '%s/transffered.png' % (opt.outf),
            normalize=True)
torch.save(cnn.state_dict(), '%s/transform_net.pth' % (opt.outf))
