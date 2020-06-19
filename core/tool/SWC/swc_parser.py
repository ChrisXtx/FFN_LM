from torch.utils import data
from typing import Sequence, List
import h5py
from core.data.utils import *
import random
import numpy as np
import skimage
from core.tool.tools import *


def swc_fusion(dir_path, save_path):

    sorted_swc = sort_files(dir_path)
    sk_id_cnt = 0
    swc_fused_data = np.zeros((1, 7))
    for swc_file in sorted_swc:
        swc_data = np.loadtxt(dir_path + swc_file)
        num_sk = swc_data.shape[0]

        swc_data[:, 0] += sk_id_cnt
        swc_data[:, 6] += sk_id_cnt
        swc_fused_data = np.concatenate((swc_fused_data,swc_data))

        print(swc_fused_data.shape)
        sk_id_cnt += num_sk

    fused_file_name = (dir_path.split("/"))[-2]
    with open(save_path + str(fused_file_name) + "_fused.swc", 'w') as swc_fused_file:
        for sk_point in swc_fused_data:
            sk_str = '{} {} {:g} {:g} {:g} {:g} {}'.format(
                sk_point[0],  sk_point[1],  sk_point[2],  sk_point[3],  sk_point[4],  sk_point[5],  sk_point[6])
            swc_fused_file.write(sk_str + '\n')









def swc_to_inf_seeds(swc, images_shape, swc_scaling, input_size, delta, buffer_dis, inf_seed_display, seed_save_path):

    """
    Finding all the seeds that are ideal for inference from skeleton points
    Avoiding seeds close to overlapping region

    :param swc:
    :param images:
    :param swc_scaling:          original / scaling
    :param input_size:           fov size of model
    :param delta:                delta size of model
    :param buffer_dis:           specify the distance to the overlapping area
    :param inf_seed_display:     inspect seeds locations
    :return:
    """

    swc_data = np.loadtxt(swc)


    swc_data = np.round(swc_data)
    sk_num = len(swc_data[:, 5])

    cube_size = input_size + 2 * delta
    rad_cube = np.array(cube_size) / 2

    inf_seed_dict = {}
    inf_seed_list = []
    for point in range(sk_num - buffer_dis - 1):

        sk_id = swc_data[point][0]
        parent = swc_data[point][6]

        for dis in range(-buffer_dis, buffer_dis + 1, 1):
            id_neighbor = swc_data[point + dis][0]
            parent_neighbor = swc_data[point + dis][6]

            if id_neighbor - parent_neighbor != 1:
                if parent == -1:
                    pass
                else:
                    p_index = int(parent) - 1
                    swc_data[p_index][1] = 0
                    print(swc_data[p_index][0])

    for point in range(sk_num):
        over_lap_sk_id = swc_data[point][1]
        if over_lap_sk_id != 0:
            sk_id = swc_data[point][0]
            x = swc_data[:, 2][point] / swc_scaling
            y = swc_data[:, 3][point] / swc_scaling
            z = swc_data[:, 4][point] / swc_scaling
            if ((z <= (images_shape[0] - rad_cube[0])) & (z >= rad_cube[0])
                    & (y <= (images_shape[1] - rad_cube[1])) & (y >= rad_cube[1])
                    & (x <= (images_shape[2] - rad_cube[2])) & (x >= rad_cube[2])):
                seed = (int(z), int(y), int(x))

                inf_seed_dict[sk_id] = seed
                inf_seed_list.append(seed)

    for coord in inf_seed_list:
        target_shape = (1, 1, 1)
        target_shape = np.array(target_shape)
        coord = np.array(coord)
        start = coord - target_shape // 2
        end = start + target_shape
        selector = [slice(s, e) for s, e in zip(start, end)]
        inf_seed_display[tuple(selector)] = 1

    with h5py.File(seed_save_path + 'seeds.h5', 'w') as f:

        f.create_dataset('seeds', data= np.array(inf_seed_list) , compression='gzip')
        f.create_dataset('inf_seed_display', data=inf_seed_display, compression='gzip')

    return inf_seed_dict, inf_seed_list

def run ():
    # script

    path = '/home/x903102883/2017EXBB/PF_inf/pf/axnol/'
    swc_fusion(path, path)
    fused_file_name = (path.split("/"))[-2]
    swc = path + str(fused_file_name) + "_fused.swc"
    image_shape = (160, 500, 500)
    inf_seed_display = np.zeros(image_shape)
    seed_save = '/home/x903102883/FFN_LM_v0.2/data/agglomeration_net_test/pf_axonal_1/'
    inf_seed_dict, inf_seed_list = swc_to_inf_seeds(swc, image_shape, 2, (39, 39, 39),
                                                    (4, 4, 4), 1, inf_seed_display,
                                                    seed_save)
if __name__ == '__main__':
    run()






