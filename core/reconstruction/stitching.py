import os
import h5py
import argparse
from core.data.utils import *
import pickle
from core.tool.tools import *
import sys
from .split_merge_reconstruction import *
sys.setrecursionlimit(10**7)


def read_the_coord(seg_data, id_stack, save_path, part):

    ids = np.unique(seg_data)

    for id in ids:
        if id == 0:
            continue
        id_mask = (seg_data == id)
        seg_coords = np.argwhere(seg_data)

        try:
            with h5py.File(save_path + "segs_coord_part{}.h5".format(part),'a') as f:
                f.create_dataset(id+id_stack, data=seg_coords, compression='gzip')
        except OSError:
            continue


read_the_coord(seg_data, id_stack, save_path, part)



