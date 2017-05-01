import cv2
import os
import numpy as np

# Get user supplied values
# Create the haar cascade
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
for fn in sorted(os.listdir('../data/CelebA/images')):
    print fn
    image = cv2.imread(os.path.abspath('../data/CelebA/images/' + fn))
    print(os.path.abspath('../data/CelebA/images/' + fn))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 5, 5)

    if len(faces) == 0:
        pass
    else:
        x,y,w,h = faces[0]
        image_crop = image[y:y+w, x:x+w, :]
        image_resize = cv2.resize(image_crop, (64, 64))

        if not os.path.exists('64_crop/'):
            os.makedirs('64_crop/')
        cv2.imwrite('64_crop/' + fn[:-4] + '_crop' + fn[-4:], image_resize)

        if not os.path.exists('128_crop/'):
            os.makedirs('128_crop/')
        image_resize = cv2.resize(image_crop, (128, 128))
        cv2.imwrite('128_crop/' + fn[:-4] + '_crop' + fn[-4:], image_resize)





