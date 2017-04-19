import torch
import random

class ImagePool(object):
    def __init__(self,poolSize):
        super(ImagePool,self).__init__()
        self.poolSize = poolSize
        if(poolSize > 0):
            self.num_imgs = 0
            self.images = []

    def Query(self,img):
        # not using lsGAN
        if(self.poolSize == 0):
            return img
        if(self.num_imgs < self.poolSize):
            # pool is not full
            self.images.append(img)
            self.num_imgs = self.num_imgs + 1
            return img
        else:
            # pool is full, by 50% chance randomly select an image tensor,
            # return it and replace it with the new tensor, by 50% return
            # the newly generated image
            p = random.random()
            if(p > 0.5):
                idx = random.randint(0,self.poolSize-1)
                tmp = self.images[idx]
                self.images[idx] = img
                return tmp
            else:
                return img
