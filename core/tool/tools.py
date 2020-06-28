
import h5py
import numpy as np
from pathlib import Path
import natsort
import skimage
import argparse
from scipy import ndimage
import pickle
import h5py
import os




def sort_files(dir_path):
    entries = Path(dir_path)
    files = []
    for entry in entries.iterdir():
        files.append(entry.name)
    sorted_files = natsort.natsorted(files, reverse=False)
    return sorted_files

def pickle_obj(obj, name, path ):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_raw_image(path):
    if path[-2:] == 'h5':
        with h5py.File(path, 'r') as f:
            images = (f['/image'][()].astype(np.float32) - 128) / 33
    else:
        images = ((skimage.io.imread(path)).astype(np.float32) - 128) / 33
    return images



def resume_dict_load(dict_path, name, resume_obj):
    if os.path.exists(dict_path + name + '.pkl'):
        resume = load_obj(dict_path + name + '.pkl')
    else:
        resume = {'resume_seed': resume_obj}
    resume_obj = resume['resume_seed']
    print('resume', resume_obj)
    return resume


def resume_re_segd_count_mask(path,shape):
    if os.path.exists(path + 're_seged_count_mask.tif'):
        re_seged_count_mask = skimage.io.imread(path + 're_seged_count_mask.tif')
    else:
        re_seged_count_mask = np.zeros(shape, dtype=np.uint8)
    return re_seged_count_mask

def seeds_to_dict(seeds_path):
    inf_seed_dict = {}
    if os.path.exists(seeds_path):
        with h5py.File(seeds_path, 'r') as segs:
            seeds = segs['seeds'][()]
            seed_id = 1
            seeds = list(seeds)
            for coord in seeds:
                inf_seed_dict[seed_id] = coord
                seed_id += 1
    else:
        print("seeds file is not exist")

    return inf_seed_dict











def z_resize_no_inter(data,factor):
    resized = skimage.transform.resize(data,
                                   (data.shape[0] * factor, data.shape[1], data.shape[2]),
                                   mode='edge',
                                   anti_aliasing=False,
                                   anti_aliasing_sigma=None,
                                   preserve_range=True,
                                   order=0)
    return resized




































"""
def increase_contrast(image):
    #Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) to increase
    #the contrast of an image. Found on Stack Overflow, written by Jeru Luke.

    # Converting image to LAB Color model
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Splitting the LAB image to different channels
    l, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to RGB model
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final



def histo_equal(img):
    '''
    This function takes an image and performs
    an adaptive histogram equalization on it
    for smoother lighting.
    :param src: The image to be smoothed
    :type src: str
    :type src: numpy.ndarray
    :return:
    '''


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    ret = clahe.apply(img)

    return ret


def calhe_3d(image):
    print(image.shape)
    stack = histo_equal(image[0])
    stack= np.expand_dims(stack, axis=0)
    print(stack.shape)
    for slice in range(image.shape[0]-1):
        slice_out = histo_equal(image[slice+1])
        slice_out = np.expand_dims(slice_out, axis=0)
        stack = np.concatenate((stack,slice_out),axis= 0)
        print(stack.shape)
    return stack
"""
