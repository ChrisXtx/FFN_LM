#import imageio
#import imgaug as ia
#from imgaug import augmenters as iaa

from torch.utils import data
from typing import Sequence, List
import h5py
import random
import numpy as np



def geometric_transform(data1, data2):

    t1 = random.choice([np.fliplr, np.flipud])
    t2 = random.choice([np.fliplr, np.flipud])
    data1 = t2(t1(data1))
    data2 = t2(t1(data2))

    return data1, data2

"""
def MHAS_transform(data1):
    z = data1.shape[0]
    degree = random.randrange(8, 12, 1)
    hue_m = degree / 10
    aug = iaa.MultiplyHueAndSaturation((hue_m, hue_m))
    for stack in range(z):
        data1[stack] = aug.augment_image(data1[stack])
    return data1


def ATHAS_transform(data1):
    z = data1.shape[0]
    para = random.randrange(-15, 15, 1)
    aug = iaa.AddToHueAndSaturation((para, para))
    for stack in range(z):
        data1[stack] = aug.augment_image(data1[stack])
    return data1
"""
