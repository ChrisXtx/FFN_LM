
import os
import h5py
import argparse
from core.models.ffn import FFN
from core.data.utils import *


parser = argparse.ArgumentParser(description='inference script')
parser.add_argument('--data', type=str, default='/home/x903102883/2017EXBB/inference_test/Fused_test_sample5/Fused-RGB-1_test_sample502down2_down_left.tif', help='input images')
parser.add_argument('--SWC', type=str, default='/home/x903102883/2017EXBB/inference_test/Fused_test_sample5/Fused_test_sample502_botleft.swc', help='SWC_skeletons')
parser.add_argument('--model', type=str, default='/home/x903102883/xtx/FFN_LM_v0.2/model/model_down2_39_10/bound_40_testffn_model_fov:39_delta:10_depth:26_recall78.44700292297159.pth', help='path to ffn model')


parser.add_argument('--data_save', type=str, default='/home/x903102883/xtx/FFN_LM_v0.2/data/agglomeration_sample/Fused_test_sample5/502_down_left/seedOnseged_ratio_0.2/', help='SWC_skeletons')
parser.add_argument('--buffer_distance', type=int, default=0, help='SWC_skeletons_dis_from_overlap')


parser.add_argument('--delta', default=(10,10,10), help='delta offset')
parser.add_argument('--input_size', default=(39, 39, 39), help ='input size')
parser.add_argument('--SWC_scaling', type=int, default=2, help='SWC_scaling')
parser.add_argument('--depth', type=int, default=26, help='depth of ffn')
parser.add_argument('--seg_thr', type=float, default=0.6, help='input size')
parser.add_argument('--mov_thr', type=float, default=0.6, help='input size')
parser.add_argument('--act_thr', type=float, default=0.8, help='input size')


args = parser.parse_args()


def run():

    model = FFN(in_channels=4, out_channels=1, input_size=args.input_size, delta=args.delta, depth=args.depth).cuda()
    assert os.path.isfile(args.model)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    if args.data[-2:] == 'h5':
        with h5py.File(args.data, 'r') as f:
            images = (f['/image'][()].astype(np.float32) - 128) / 33
            # labels = g['label'].value
    else:
        images = ((skimage.io.imread(args.data)).astype(np.float32) - 128) / 33





    # swc_seed_generation_strategy
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

    inference_seed_dict = {}
    inference_seed_list = []
    for point in range(dataNum):
        over_lap_id = swc_data[point][1]
        if over_lap_id != 0:
            print(swc_data[point][0])
            id = swc_data[point][0]
            x =swc_data[:, 2][point]/args.SWC_scaling
            y =swc_data[:, 3][point]/args.SWC_scaling
            z =swc_data[:, 4][point]/args.SWC_scaling
            if   ((z <= (images_shape[0]-rad_cube[0]))&(z >= rad_cube[0])
                &(y <= (images_shape[1]-rad_cube[1]))&(y >= rad_cube[1])
                &(x <= (images_shape[2]-rad_cube[2]))&(x >= rad_cube[2])):
                seed = (int(z), int(y), int(x))
                print("id, seed",id,seed)
                inference_seed_dict[id] = seed
                inference_seed_list.append(seed)

    print (inference_seed_list)
    print (inference_seed_dict)

    inference_seed_display = np.zeros(images.shape[:-1])

    for coord in inference_seed_list:
        target_shape = (1, 1, 1)
        target_shape = np.array(target_shape)
        coord = np.array(coord)
        start = coord - target_shape // 2
        end = start + target_shape
        selector = [slice(s, e) for s, e in zip(start, end)]
        inference_seed_display[tuple(selector)] = (255)

    print("num_of_coords",len(inference_seed_list))
    with h5py.File(args.data[:-3]+"h5", 'w') as f:

        f.create_dataset('image_raw', data=images, compression='gzip')
        f.create_dataset('inference_coor', data=inference_seed_list, compression='gzip')
        f.create_dataset('inference_seed_display', data=inference_seed_display, compression='gzip')


    canva = Canvas(model, images, inference_seed_list, args.input_size, args.delta, args.seg_thr, args.mov_thr, args.act_thr, args.data_save)
    id = 0

    for key in inference_seed_dict.keys():
        pos = inference_seed_dict[ key]
        if canva.segment_at(pos,key) == True:
            continue



if __name__ == '__main__':
    run()
