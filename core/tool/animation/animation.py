
from torch.utils import data
from typing import Sequence, List
import h5py

import random
import numpy as np
from imgaug import augmenters as iaa
import skimage




def abs_gap_aug(data,freq,redu_range):

    z = int(data.shape[0])
    y = int(data.shape[1])
    x = int(data.shape[2])
    radi_R = int(data.shape[0]/4)

    num_gap = random.randrange(0, freq, 1)
    aug_list= [[104,113,120],[104,94,84],[114,74,112]]
    for coord in range(len(aug_list)):
        #coordz = random.randrange(0, z, 1)
        #coordy = random.randrange(0, y, 1)
        #coordx = random.randrange(0, x, 1)
        #radi = random.randrange(0, radi_R, 1)
        radi = 5
        coordz = aug_list[coord][0]
        coordy = aug_list[coord][1]
        coordx = aug_list[coord][2]
        print("location",coordz,coordy,coordx)
        for xd in range(-radi, radi + 1):
            #radi = random.randrange(0, radi_R, 1)
            for yd in range(-radi, radi + 1):
                #radi = random.randrange(0, radi_R, 1)
                for zd in range(-radi, radi + 1):
                    if ((coordz + zd)>=data.shape[0])|((coordz + zd) <=0)|((coordy + yd)>=data.shape[1])|((coordy + yd) <=0)|((coordx + xd)>=data.shape[2])|((coordx + xd) <=0):
                        continue

                    temp =  int(redu_range*10)
                    redu = random.randrange(0, temp, 1)
                    print(coordz + zd,coordy + yd,coordx + xd)
                    rand = random.randrange(0, 10, 1)
                    if (rand == 7)|(rand == 6)|(rand == 9)|(rand == 8):
                        data[coordz + zd, coordy + yd, coordx + xd] = (data[coordz + zd, coordy + yd, coordx + xd])*0.55
                        continue
                    data[coordz + zd, coordy + yd, coordx + xd] = (data[coordz + zd, coordy + yd, coordx + xd])*0.4



with h5py.File('image-2.h5', 'r') as raw:
        data= (raw['image'][()])  # .a
print(data)

skimage.io.imsave('test2.tif', data.astype('uint8'))

data1 = data.copy()
print(data1==data)
abs_gap_aug(data1,5,0.5)
print("!!!!!!!!!!!!!!!")
skimage.io.imsave('test1.tif', data1.astype('uint8'))
print(data1==data)