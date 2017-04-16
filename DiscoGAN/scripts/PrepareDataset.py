# We use dataset from pix2pix. This is a paired dataset, so we need to split
# each pair and generate unpaired dataset

import argparse
import os
from PIL import Image
from glob import glob

parser = argparse.ArgumentParser(description='split paired dataset to unpaired dataset')
parser.add_argument('--dataPath', default='../facades', help='path to dataset folder')

########   make A and B folder   ########
opt = parser.parse_args()
paths = glob(opt.dataPath+'/*')
for path in paths:
    if not os.path.exists(os.path.join(path,'A')):
        os.makedirs(os.path.join(path,'A'))
    if not os.path.exists(os.path.join(path,'B')):
        os.makedirs(os.path.join(path,'B'))

########   split images and put them into corresponding folders   ########
def isImg(filename):
    ext = filename.split('.')[-1]
    if(ext in ['png','jpg','jpeg','ppm']):
        return True

def split(imgPath):
    img = Image.open(imgPath)
    w,h = img.size
    imgA = img.crop((0, 0, w/2, h))
    imgB = img.crop((w/2, 0, w, h))
    return imgA, imgB

for path in paths:
    print('Processing folder: '+path)
    for fn in os.listdir(path):
        if(isImg(fn)):
            imgA,imgB = split(os.path.join(path,fn))
            imgA.save(os.path.join(path,'A',fn))
            imgB.save(os.path.join(path,'B',fn))

choice = raw_input('Delete original paired data? (y/n) WARNING: DELETED DATASET IS UNRECOVERABLE')
if(choice == 'y'):
    for path in paths:
        for fn in os.listdir(path):
            if(isImg(fn)):
                os.system('rm '+os.path.join(path,fn))
