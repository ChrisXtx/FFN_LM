
import h5py
import numpy as np

from pathlib import Path
import natsort
import skimage
import argparse
from scipy import ndimage


def z_resize_no_inter(data,factor):
    resized = skimage.transform.resize(data,
                                   (data.shape[0] * factor, data.shape[1], data.shape[2]),
                                   mode='edge',
                                   anti_aliasing=False,
                                   anti_aliasing_sigma=None,
                                   preserve_range=True,
                                   order=0)
    return resized



def sort_files(dir_path):
    entries = Path(dir_path)
    files = []
    for entry in entries.iterdir():
        files.append(entry.name)
    sorted_files = natsort.natsorted(files, reverse=False)
    return sorted_files

"""
def increase_contrast(image):
    """Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) to increase
    the contrast of an image. Found on Stack Overflow, written by Jeru Luke."""

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
