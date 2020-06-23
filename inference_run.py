import os
import h5py
import argparse
from core.models.ffn import FFN
import core.data.utils as inf
from core.tool.tools import *
import threading
import torch
import random

parser = argparse.ArgumentParser(description='inference script')
parser.add_argument('--data', type=str,
                    default='/home/x903102883/2017EXBB/whole_volume_inf/Fused-RGB_down2.tif',
                    help='input images')
parser.add_argument('--seed', type=str,
                    default='/home/x903102883/2017EXBB/whole_volume_inf/part4/whole_part4_seeds.h5',
                    help='swc_skeletons')
parser.add_argument('--model', type=str,
                    default='/home/x903102883/2017EXBB/whole_volume_inf/down_2_adamffn_model_fov_39_delta_4_depth_26_recall87.6557408472302.pth',
                    help='path to ffn model')

parser.add_argument('--data_save', type=str,
                    default='/home/x903102883/2017EXBB/whole_volume_inf/part4/',
                    help='swc_skeletons')

parser.add_argument('--threads', type=int, default=1, help='tag the files')
parser.add_argument('--save_chunk', type=int, default=5000, help='separate the seg_coords from seeds by chunk')
parser.add_argument('--buffer_distance', type=int, default=0, help='swc_skeletons_dis_from_overlap')
parser.add_argument('--delta', default=(4, 4, 4), help='delta offset')
parser.add_argument('--input_size', default=(39, 39, 39), help='input size')
parser.add_argument('--swc_scaling', type=int, default=2, help='swc_scaling')
parser.add_argument('--depth', type=int, default=26, help='depth of ffn')
parser.add_argument('--seg_thr', type=float, default=0.6, help='input size')
parser.add_argument('--mov_thr', type=float, default=0.8, help='movable thr')
parser.add_argument('--act_thr', type=float, default=0.8, help='activation of seg')
parser.add_argument('--re_seg_thr', type=int, default=2, help='will not seed here if segmented many times')
parser.add_argument('--vox_thr', type=int, default=500, help='remove if too small')
parser.add_argument('--resume_seed', type=int, default=94409, help='resume_seed')
parser.add_argument('--tag', type=str, default='whole_par4', help='tag the files')

args = parser.parse_args()

# pre_load

if args.data[-2:] == 'h5':
    with h5py.File(args.data, 'r') as f:
        images = (f['/image'][()].astype(np.float32) - 128) / 33
else:
    images = ((skimage.io.imread(args.data)).astype(np.float32) - 128) / 33

segs_saved_files = sort_files(args.data_save)
files_num = len(segs_saved_files)-1
if files_num > 1:
    resume_segs_file = ''
    # find the last part
    for file_index in range(files_num):
        if 'seg_of_seeds' not in segs_saved_files[files_num]:
            files_num -= 1

            continue
        resume_segs_file = segs_saved_files[files_num]

    with h5py.File(args.data_save + resume_segs_file, 'r') as segs:
        ids = list(segs.keys())
        sorted_ids = natsort.natsorted(ids, reverse=False)
        resume_seed = int(sorted_ids[len(sorted_ids)-1])
    args.resume_seed = resume_seed


def canvas_init(process_id):
    model = FFN(in_channels=4, out_channels=1, input_size=args.input_size, delta=args.delta, depth=args.depth).cuda()
    assert os.path.isfile(args.model)
    model.load_state_dict(torch.load(args.model))
    model.eval()


    canvas_inf = inf.Canvas(model, args.input_size, args.delta, args.seg_thr, args.mov_thr,
                        args.act_thr, args.re_seg_thr, args.vox_thr, args.data_save, process_id)
    inf_seed_dict = {}
    with h5py.File(args.seed, 'r') as segs:
        seeds = segs['seeds'][()]
        seed_id = 1
        random.seed(30)
        seeds = list(seeds)
        for coord in seeds :

            inf_seed_dict[seed_id] = coord
            seed_id += 1

    return canvas_inf, inf_seed_dict

def run (canvas_inf, inf_seed_dict, process_id, process_num):

    # run inference on every seed and save their segmentation
   

    # individualize the dict
    ps_spe = 'inf_seed_dict' + str(process_id)
    multi_th_inf_seed_dict = {}
    multi_th_inf_seed_dict[ps_spe] = inf_seed_dict

    for seed_id in multi_th_inf_seed_dict[ps_spe].keys():
        if not seed_id % process_num == process_id:
            continue
        if seed_id < args.resume_seed:
            continue
        coord = inf_seed_dict[seed_id]
        coord = tuple(coord)
        if coord[1] < 10 | coord[1] > 120:
            continue

        if canvas_inf.segment_at(coord, seed_id, args.tag):
            
            continue





if __name__ == '__main__':
    # multiprocess code
   

    threads = args.threads

    for thread in range(threads):
        canvas_inf, inf_seed_dict = canvas_init(thread)
        convas_thread = threading.Thread(target=run, args=(canvas_inf, inf_seed_dict, thread, threads))  # main process
        convas_thread.start()
