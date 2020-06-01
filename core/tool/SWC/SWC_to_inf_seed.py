

import os
import h5py
import argparse
from core.models.ffn import FFN
from core.data.utils import *

parser = argparse.ArgumentParser(description='inference script')
#parser.add_argument('--data', type=str, default='/home/x903102883/FFN_LM_v0.2/core/tool/SWC/(0, 1024, 0)_raw.tif', help='input images')
parser.add_argument('--SWC', type=str, default='/home/x903102883/2017EXBB/2017EXBB_SWC/tile5.swc', help='SWC_skeletons')
parser.add_argument('--buffer_distance', type=int, default=3, help='SWC_skeletons')
parser.add_argument('--delta', default=(15,15, 15), help='delta offset')
parser.add_argument('--input_size', default=(51,51,51), help='input size')
parser.add_argument('--resume', type=str, default= '/home/x903102883/2017EXBB/2017EXBB_SWC/2017EXBB_inf_seed.h5', help='resume')


args = parser.parse_args()


def run():


    images = np.zeros((3773,1549,160))

    #args.resume = None
    if args.resume is not None:
        with h5py.File(args.resume, 'r') as f:
                inference_seed_list  = f['inference_coor'][()]
                inference_seed_list = list(inference_seed_list)
        print("resume",len(inference_seed_list))
    else:
        inference_seed_list = []

    # swc_seed strategy
    swc_data = np.loadtxt(args.SWC)
    images_shape = images.shape

    #swc_overlap_mask = np.ones((images_shape[0], images_shape[1], images_shape[2]), dtype=bool)
    expansion = 1
    swc_data[:, 2] = swc_data[:, 2] * expansion
    swc_data[:, 3] = swc_data[:, 3] * expansion
    swc_data[:, 4] = swc_data[:, 4] * expansion
    swc_data[:, 5] = swc_data[:, 5] * expansion
    swc_data = np.round(swc_data)
    swc_data[:, 5] = swc_data[:, 5]

    dataNum = len(swc_data[:, 5])

    input_size =np.array(args.input_size)
    delta = np.array(args.delta)
    cube_size = input_size + 2 * delta
    rad_cube = np.array(cube_size)/2
    print("rad_cube", rad_cube)


    buffer_dis = args.buffer_distance # the skeleton distance to the overlapping area

    for point in range(dataNum - buffer_dis -1):

        id = swc_data[point][0]
        parent = swc_data[point][6]

        for dis in range(-buffer_dis,buffer_dis+1,1):
            id_neighbor =  swc_data[point+dis][0]
            parent__neighbor  = swc_data[point+dis][6]

            if id_neighbor - parent__neighbor != 1:
                print("run_inf_id", id)
                print("run_inf_dif", id - parent)

                if parent == -1:
                    pass
                else:
                    p_index= int(parent)-1
                    swc_data[p_index][1] = 0
                    print(swc_data[p_index][0])


    #inference_seed_dict = {}
    offset = 8000

    for point in range(dataNum):
        over_lap_id = swc_data[point][1]
        if over_lap_id != 0:
            print(swc_data[point][0])
            id = swc_data[point][0]
            x =swc_data[:, 2][point]
            y =swc_data[:, 3][point] + offset
            z =swc_data[:, 4][point]
            if   ((x <= (images_shape[0]-rad_cube[0]))&(x >= rad_cube[0])
                &(y <= (images_shape[1]-rad_cube[1] +offset) )&(y >= (rad_cube[1]+offset))
                &(z <= (images_shape[2]-rad_cube[2]))&(z >= rad_cube[2])):
                seed = (int(z), int(y), int(x))
                print("id, seed",id,seed)
                #inference_seed_dict[id] = seed
                inference_seed_list.append(seed)

    print (inference_seed_list)
    #print(inference_seed_dict)

    inference_seed_display = np.zeros(images.shape)

    for coord in inference_seed_list:
        target_shape = (3, 3, 3)
        target_shape = np.array(target_shape)
        coord = np.array(coord)
        start = coord - target_shape // 2
        end = start + target_shape
        selector = [slice(s, e) for s, e in zip(start, end)]
        inference_seed_display[tuple(selector)] = (255)


    print("num_of_coords",len(inference_seed_list))

    with h5py.File("/home/x903102883/2017EXBB/2017EXBB_SWC/2017EXBB_inf_seed.h5", 'w') as f:

        f.create_dataset('inference_coor', data=inference_seed_list, compression='gzip')
        #f.create_dataset('inference_seed_display', data=inference_seed_display, compression='gzip')







if __name__ == '__main__':
    run()
