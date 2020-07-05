import os
import h5py
import argparse
import pickle
from core.tool.tools import *
import skimage
import random
import argparse
from core.reconstruction.split_merge_reconstruction import *
import sys

sys.setrecursionlimit(10 ** 7)

parser = argparse.ArgumentParser(description='inference script')

parser.add_argument('--recon_dir', type=str,
                    default='/home/x903102883/2017EXBB/inference_agglomeration_test/shubhra_test2/test/',
                    help='input images')
parser.add_argument('--merge_ratio', type=float, default=0.1, help='input images')
parser.add_argument('--vox_thr', type=int, default=10, help='input images')
parser.add_argument('--RGB', type=bool, default=True, help='generate RGB')
parser.add_argument('--image_shape', default=(160, 128, 128), help='generate RGB')

args = parser.parse_args()


def run():
    seg_s_path_test = args.recon_dir
    merge_dict_save_path_test = seg_s_path_test
    merge_dict_path_test = seg_s_path_test + 'merge_dict.pkl'

    # segmentation: coordinates to codes
    seg_s_code_dict = convert_coor_all(seg_s_path_test)

    # pair-wise merge check
    merge_dict = merge(seg_s_path_test, seg_s_code_dict, args.merge_ratio, args.vox_thr)
    print(len(merge_dict))
    # save merge dict
    pickle_obj(merge_dict, 'merge_dict', merge_dict_save_path_test)
    merge_dict_test = load_obj(merge_dict_path_test)

    # recursively add all merged segmentation into the same group
    merge_group_dict_test = merge_seg_s(merge_dict_test)

    # reconstruction
    image_shape = args.image_shape
    segmentation = seg_s_reconstructor(seg_s_path_test, merge_group_dict_test, image_shape, cons_thr=1)
    seg_save_path = merge_dict_save_path_test + 'recon_test.tif'
    skimage.io.imsave(seg_save_path, segmentation.astype('uint32'))

    if args.RGB:
        RGB_img = seg_s_to_RGB(segmentation)
        save_path = merge_dict_save_path_test + 'recon_RGB.tif'
        skimage.io.imsave(save_path, RGB_img.astype('uint8'))


if __name__ == '__main__':
    run()
