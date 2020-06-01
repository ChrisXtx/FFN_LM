from torch.utils import data
from typing import Sequence, List
import h5py
#from data.core.utils import *
import random
import  numpy as np
import skimage

bed = np.zeros((750,750,750))
with h5py.File('comp_data_raw3_exp750.h5', 'r') as raw:
        coords = raw['coor'][()]
                #self.seed.append(logit(np.full(list(raw['label'][()].shape), 0.05, dtype=np.float32)))
        #coords = np.random.shuffle(coords)
        print(coords)

        coordscnt = 0
        for coord in coords:
                target_shape= (2,2,2)
                target_shape = np.array(target_shape)
                coord = np.array(coord)
                start = coord -target_shape // 2
                end = start + target_shape

                selector = [slice(s, e) for s, e in zip(start, end)]
                #cropped = bed[tuple(selector)]
                #mask = cropped
                bed[tuple(selector)] = (250)
                #bed[cropped] = 250
                #print(bed[coord])
                coordscnt+=1
                if  coordscnt>= 10000:
                        break


skimage.io.imsave("bed2020413.tif", bed.astype('uint8'))