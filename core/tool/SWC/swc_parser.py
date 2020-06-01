from torch.utils import data
from typing import Sequence, List
import h5py
#from data.core.utils import *
import random
import  numpy as np
import skimage



swc_data = np.loadtxt('raw3_swc.swc')
print(swc_data[0])
expansion = 3

swc_data[:,2] = swc_data[:,2]*expansion
swc_data[:,3] = swc_data[:,3]*expansion
swc_data[:,4] = swc_data[:,4]*expansion
swc_data[:,5] = swc_data[:,5]*expansion

swc_data= np.round(swc_data)
swc_data[:,5] = swc_data[:,5]+ 6


bed = np.zeros((1500,750,750))
dataNum = len(swc_data[:,5])
print(dataNum)

coord_to_data_dict = {}


for point in range(dataNum):
    x = swc_data[:, 2][point]
    y = swc_data[:, 3][point]
    z = swc_data[:, 4][point]
    rad = swc_data[:, 5][point]
    rad = int(rad)

    strx = str(x)
    stry = str(y)
    strz = str(z)
    coord = strx + stry + strz
    #print("*********************************")
    #print(coord)
    #print(swc_data[point])


    coord_to_data_dict.update({coord : swc_data[point]})
    print(coord_to_data_dict[coord])

    if (x <= 0) | (y <= 0) | (z <= 0) |(z >= (bed.shape[0]-rad)) |(y >= (bed.shape[1]-rad))|(x >= (bed.shape[2]-rad)):
        continue

    if rad % 2 ==1:
        rad +=1
    target_shape = (rad, rad, rad)
    target_shape = np.array(target_shape)
    coord = (int(z),int(y),int(x))
    coord = np.array(coord)
    start = coord - target_shape // 2
    end = start + target_shape

    selector = [slice(s, e) for s, e in zip(start, end)]
    print(selector)
    bed[tuple(selector)] = swc_data[:, 0][point]
    #bed[int(z),int(y),int(x)] = 250


skimage.io.imsave('bed.tif', bed.astype('uint32'))


