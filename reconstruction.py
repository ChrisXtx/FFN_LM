
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
sys.setrecursionlimit(10**7)



parser = argparse.ArgumentParser(description='inference script')
parser.add_argument('--recon_dir', type=str,
                    default='/home/x903102883/2017EXBB/PF_inf/pf/axonal2/opt_test/',
                    help='input images')
parser.add_argument('--merge_ratio', type=float, default=0.2, help='input images')
parser.add_argument('--vox_thr', type=int, default=100, help='input images')
parser.add_argument('--RGB', type=bool, default=False, help='generate RGB')
parser.add_argument('--image_shape', default=(160, 500, 500), help='generate RGB')

args = parser.parse_args()

def run():
    segs_path_test = args.recon_dir
    merge_dict_save_path_test = segs_path_test
    merge_dict_path_test = segs_path_test + 'merge_dict.pkl'

    segs_code_dict = convert_coor_all(segs_path_test)
    merge_dict = merge(segs_path_test, segs_code_dict, args.merge_ratio, args.vox_thr)
    print(len(merge_dict))

    pickle_obj(merge_dict, 'merge_dict', merge_dict_save_path_test)
    merge_dict_test = load_obj(merge_dict_path_test)

    merge_group_dict_test = merge_segs(merge_dict_test)

    image_shape = args.image_shape

    segmentation = segs_reconstructor(segs_path_test, merge_group_dict_test, image_shape, cons_thr=1)

    seg_save_path = merge_dict_save_path_test + 'recon_segs.tif'
    skimage.io.imsave(seg_save_path, segmentation.astype('uint32'))

    if args.RGB:
        RGB_img = segs_to_RGB(segmentation)
        save_path = merge_dict_save_path_test + 'test_RGB.tif'
        skimage.io.imsave(save_path, RGB_img.astype('uint8'))



if __name__ == '__main__':
    run()
