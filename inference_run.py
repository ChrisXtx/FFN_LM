import os
import h5py
import argparse
from core.models.ffn import FFN
from core.tool.tools import *
import threading
import torch
import random
import core.inference.inference as inf

parser = argparse.ArgumentParser(description='inference script')
parser.add_argument('--data', type=str,
                    default='/home/x903102883/2017EXBB/PF_inf/pf/axonal2/pf_axonal_2_raw_down2.tif',
                    help='input images')

parser.add_argument('--seed', type=str,
                    default='/home/x903102883/2017EXBB/PF_inf/pf/axonal2/axonal2_seeds.h5',
                    help='swc_based_seeds')

parser.add_argument('--model', type=str,
                    default='/home/x903102883/FFN_LM_v0.2/model/down_2_adamffn_model_fov:39_delta:4_depth:26_recall93.56709552696488.pth',
                    help='path to ffn model')

parser.add_argument('--data_save', type=str,
                    default='/home/x903102883/2017EXBB/PF_inf/pf/axonal2/',
                    help='save_result')

parser.add_argument('--save_chunk', type=int, default=5000, help='separate the seg_coords from seeds by chunk')
parser.add_argument('--resume_seed', type=int, default=0, help='resume_seed')
parser.add_argument('--tag', type=str, default='whole_par4', help='tag the files')


parser.add_argument('--threads', type=int, default=1, help='tag the files')
parser.add_argument('--delta', default=(4, 4, 4), help='delta offset')
parser.add_argument('--input_size', default=(39, 39, 39), help='input size')
parser.add_argument('--depth', type=int, default=26, help='depth of ffn')

parser.add_argument('--seg_thr', type=float, default=0.6, help='input size')
parser.add_argument('--mov_thr', type=float, default=0.8, help='movable thr')
parser.add_argument('--act_thr', type=float, default=0.8, help='activation of seg')
parser.add_argument('--re_seg_thr', type=int, default=3, help='will not seed here if segmented many times')
parser.add_argument('--vox_thr', type=int, default=500, help='remove if too small')
parser.add_argument('--manual_seed', type=bool, default=False, help='specify the seeds source')


args = parser.parse_args()


images = load_raw_image(args.data)
resume = resume_dict_load(args.data_save, 'resume_seed', args.resume_seed)
re_seged_count_mask = resume_re_segd_count_mask(args.data_save, images.shape[:-1])


def canvas_init(process_id):

    model = FFN(in_channels=4, out_channels=1, input_size=args.input_size, delta=args.delta, depth=args.depth).cuda()

    assert os.path.isfile(args.model)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    canvas_inf = inf.Canvas(model, images, args.input_size, args.delta, args.seg_thr, args.mov_thr,
                            args.act_thr, args.re_seg_thr, args.vox_thr, args.data_save, re_seged_count_mask,
                            args.save_chunk, args.resume_seed, args.manual_seed, process_id)

    inf_seed_dict = {}
    if os.path.exists(args.seed):
      inf_seed_dict = seeds_to_dict(args.seed)

    return canvas_inf, inf_seed_dict


def inf_run(canvas, inf_seed, process_id, process_num):
    # run inference on every seed and save their segmentation

    for seed_id in inf_seed.keys():
        if not seed_id % process_num == process_id:
            continue
        if seed_id < args.resume_seed:
            continue
        coord = inf_seed_dict[seed_id]
        coord = tuple(coord)
        if coord[1] < 10 | coord[1] > 130:
            continue
        if canvas.segment_at(coord, seed_id, args.tag):
            if seed_id % 10 == 0:
                resume['resume_seed'] = seed_id
                pickle_obj(resume, 'resume_seed', args.data_save)
            continue


if __name__ == '__main__':
    # multiprocess code

    threads = args.threads

    for thread in range(threads):
        canvas_inf, inf_seed_dict = canvas_init(thread)
        # single_seed_run
        #inf_seed_dict = {1: [z,y,x]}

        convas_thread = threading.Thread(target=inf_run, args=(canvas_inf, inf_seed_dict, thread, threads))  # main process
        convas_thread.start()
