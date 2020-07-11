
from torch.utils import data
from typing import Sequence, List
import h5py
from core.data.utils import *
import random
import numpy as np
import skimage
from core.tool.tools import *


print(type((128,128,3)))
img_dir = '/home/x903102883/2017EXBB/train_down2_pad30_bound100/train_down2/'

def stack_imgs(img_dir, save_path, size, pad, obj, type_spe, coor=False):
    files = sort_files(img_dir)

    stack = np.zeros(size).astype(type_spe)
    coor_stack = np.array([[100,100,100]])

    stack = np.expand_dims(stack, axis=0)
    stack_cnt = 1
    if len(size) > 2:
        padder = np.zeros((pad,size[0],size[1],size[2])).astype(type_spe)
    else:
        padder = np.zeros((pad, size[0], size[1])).astype(type_spe)

    for file in files:
        print(stack.shape)
        with h5py.File(img_dir + file, 'r') as f:
            image = (f[obj][()]).astype(type_spe)
            coors = f['/coor'][()]
            print(coors.shape)

        coors[:, 0] += stack_cnt
        print(image.shape)
        coor_stack = np.concatenate((coor_stack, coors), axis=0)
        stack = np.concatenate((stack, image), axis=0).astype(type_spe)
        stack = np.concatenate((stack, padder), axis=0).astype(type_spe)
        stack_cnt = (stack.shape)[0]
        print(file)


    return stack, coor_stack

stack_image, coor_stack_test = stack_imgs(img_dir,img_dir,(218,218,3), 50, '/image', 'uint8')
stack_label, coors_feide= stack_imgs(img_dir,img_dir,(218,218), 50, '/label', 'uint16')
seed_display = np.zeros(stack_label.shape).astype('uint8')

for coord in coor_stack_test:
    target_shape = (1, 1, 1)
    target_shape = np.array(target_shape)
    coord = np.array(coord)
    start = coord - target_shape // 2
    end = start + target_shape
    selector = [slice(s, e) for s, e in zip(start, end)]
    seed_display[tuple(selector)] = (255)



with h5py.File(img_dir + 'test1.h5', 'w') as f:

    f.create_dataset('image', data=stack_image.astype('uint8'), compression='gzip')
    f.create_dataset('label', data=stack_label, compression='gzip')
    f.create_dataset('coor', data=coor_stack_test, compression='gzip')
    f.create_dataset('seed_display', data=seed_display, compression='gzip')

