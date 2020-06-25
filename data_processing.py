import cv2
import numpy as np
import argparse
from collections import defaultdict
import h5py
import numpy as np
import tifffile
import skimage
import scipy.ndimage
from pathlib import Path
import skimage
import numpy as np
import os
import cv2
import shutil
import natsort
import argparse
import random
import pickle
import random
import matplotlib.pyplot as plt
from PIL import Image
from core.tool.tools import  sort_files





#  z y x  in numpy    xyz in imageJ

def out_of_range(location, shape, cube_rad):
    for (axis, bound) in zip(location, shape):
        if (axis > bound - cube_rad - 2) or (axis < cube_rad + 3):
            return True
        else:
            continue
    return False




def seed_scanner(label, shape, cube_rad, seed_coor_list, check_rad, safe_rad, seed_display):
    z, y, x = shape

    for z_i in range(int(z / sparsity)):
        for y_i in range(int(y / sparsity)):
            for x_i in range(int(x / sparsity)):
                location_i = [z_i * sparsity, y_i * sparsity, x_i * sparsity]
                if (label[location_i[0]][location_i[1]][location_i[2]] == 0) or (out_of_range(location_i, shape, cube_rad)):
                    continue
                if obj_diversity_check(label, location_i, check_rad, safe_rad):
                    seed_coor_list.append(location_i)
                    seed_display[location_i[0]][location_i[1]][location_i[2]] = 200


def obj_diversity_check(label, location_o, check_rad, safe_rad):

    low = np.array(location_o) - check_rad
    high = np.array(location_o) + check_rad + 1
    sel = [slice(s, e) for s, e in zip(low, high)]
    seed_cube = label[tuple(sel)]
    seed_cube_vol = int(((check_rad * 2 + 1) ** 3))

    safe_low = np.array(location_o) - safe_rad
    safe_high = np.array(location_o) + safe_rad + 1
    safe_sel = [slice(s, e) for s, e in zip(safe_low, safe_high)]
    safe_seed_cube = label[tuple(safe_sel)]
    safe_seed_cube_vol = int(((safe_rad * 2 + 1) ** 3))

    seed_loc_id = label[location_o[0]][location_o[1]][location_o[2]]

    id, count = np.unique(seed_cube, return_counts=True)
    id_dic = dict(zip(id, count))

    safe_id, safe_count = np.unique(safe_seed_cube, return_counts=True)
    safe_id_dic = dict(zip(safe_id, safe_count))

    if (len(id) < 3) \
            or (id_dic[seed_loc_id] < (seed_cube_vol * 0.0002)) \
            or (id_dic[seed_loc_id] > (seed_cube_vol * 0.3)) \
            or (id_dic[0] > (seed_cube_vol * 0.95)) \
            or (safe_id_dic[seed_loc_id] < (safe_seed_cube_vol * 0.6)):
        return False

    return True


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value




abs_path_label = '/home/x903102883/2017EXBB/16bit/labels_tiff/labels_down_2_no_avg_no_interpolation/'
abs_path_raw = '/home/x903102883/2017EXBB/16bit/raw_tiff/downsample_factor_2/down_2_constrain_avg/'
abs_path_training_data = '/home/x903102883/2017EXBB/train_down2_pad30_bound100/'



sorted_files_label = sort_files(abs_path_label)
sorted_files_raw = sort_files(abs_path_raw )

train_data_dict = dict(zip(sorted_files_raw, sorted_files_label))
cube_rad = 50
sparsity = 1
safe_rad = 2
check_rad = 12
for file in train_data_dict.keys():
    padding = 30
    seed_coor_list_i = []
    file_label = abs_path_label + train_data_dict[file]
    img_label = tifffile.TiffFile(file_label).asarray()

    img_label = np.pad(img_label, padding, pad_with, padder=0)
    seed_display = np.zeros(img_label.shape)

    seed_scanner(img_label, img_label.shape, cube_rad, seed_coor_list_i, check_rad, safe_rad, seed_display)
    seed_coor_list_np = np.array(seed_coor_list_i)


    file_raw = abs_path_raw + file
    img_raw = tifffile.TiffFile(file_raw).asarray()
    pad_config = ((padding, padding), (padding, padding), (padding, padding), (0, 0))
    img_raw = np.pad(img_raw, pad_width=pad_config, mode='constant', constant_values=0)

    train_data_file_name = abs_path_training_data + 'compsite_' + file[10:-4] + ".h5"

    print(train_data_file_name, "coords:", len(seed_coor_list_i))
    if len(seed_coor_list_i) < 300:
        continue

    with h5py.File(train_data_file_name, 'w') as f:

        f.create_dataset('image', data=img_raw, compression='gzip')
        f.create_dataset('label', data=img_label, compression='gzip')
        f.create_dataset('coor', data=seed_coor_list_np, compression='gzip')
        f.create_dataset('seed_display', data=seed_display, compression='gzip')

