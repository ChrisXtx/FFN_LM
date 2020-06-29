import os
import h5py
import argparse
from core.data.utils import *
import pickle
from core.tool.tools import *
import sys
#from .split_merge_reconstruction import *

sys.setrecursionlimit(10**7)

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--seg_data', type=str, default='/home/x903102883/Desktop/inf_whole/part9_segs.tif', help='path of seg_data')
parser.add_argument('--part', type=int, default=9, help='path of seg_data')
parser.add_argument('--save_path', type=str, default='/home/x903102883/Desktop/inf_whole/temp_coords_save/', help='save_path')

args = parser.parse_args()

def read_the_coord(seg_data, save_path, part):

    images = (skimage.io.imread(seg_data))
    id_stack = part*40000
    ids = np.unique(images)
    print('num_of_obj', len(ids))
    for id in ids:
        if id == 0:
            continue
        id_mask = (images == id)
        seg_coords = (np.argwhere(id_mask)).astype('int16')
        print(seg_coords.shape)
        print(seg_coords.dtype)

        try:
            with h5py.File(save_path + "segs_coord_part{}.h5".format(part), 'a') as f:
                f.create_dataset(str(id+id_stack), data=seg_coords, compression='gzip')
        except OSError:
            continue


read_the_coord(args.seg_data, args.save_path, args.part)


