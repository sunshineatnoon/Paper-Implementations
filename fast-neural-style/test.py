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
from utils.transformer import TransformerNet 
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--transform_net', default='', help="path to the transformer net")
parser.add_argument('--outf', default='images/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument("--content_image", default="images/dancing.jpg", help='path to style image')

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

###########   Load content image   ###########
transform = transforms.Compose([
    transforms.Scale(opt.imageSize),
    transforms.CenterCrop(opt.imageSize),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],std=[1,1,1]),
    transforms.Lambda(lambda x: x.mul_(255)),
    ])

def load_image(path,style=False):
    img = Image.open(path)
    img = Variable(transform(img))
    img = img.unsqueeze(0)
    return img

contentImg = load_image(opt.content_image) # 1x3x512x512
###########   Load netG   ###########
assert opt.transform_net != '', "transformer net must be provided!"
cnn = TransformerNet()
cnn.load_state_dict(torch.load(opt.transform_net))

if(opt.cuda):
    cnn.cuda()
    contentImg = contentImg.cuda()

transffered = cnn(contentImg)
transffered = transffered.clamp(0,255)
vutils.save_image(transffered.data,
            '%s/fast_neural_transfer.png' % (opt.outf),
            normalize=True)
