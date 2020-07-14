"""
Runs inference.
"""
import os
import argparse
from typing import Dict, Tuple
from core.models.ffn import FFN
import core.tool.tools as tools
import threading
import torch
import core.inference.inference as inf


def parse_args() -> argparse:
    """
        Parses the inference arguments.
    """
    parser = argparse.ArgumentParser(description='Inference Arguments Parser')

    parser.add_argument('--data', type=str, required=True,
                        help='Path to input images')

    parser.add_argument('--seed', type=str, default='',
                        help='seeds coordinates')

    parser.add_argument('--model', type=str, required=True,
                        help='Path to the FFN model')

    parser.add_argument('--data_save', type=str, required=True,
                        help='Path where inference result will be saved')

    parser.add_argument('--save_chunk', type=int, default=5000,
                        help='Chunk the result segmentation')

    parser.add_argument('--resume_seed', type=int, default=0,
                        help='Resume Seed')

    parser.add_argument('--tag', type=str, default='test_',
                        help='Tag info')

    parser.add_argument('--threads', type=int, default=1,
                        help='Multi-threads')

    parser.add_argument('--delta', default=(4, 4, 4),
                        help='Delta offset')

    parser.add_argument('--input_size', default=(39, 39, 39),
                        help='input size')

    parser.add_argument('--depth', type=int, default=26,
                        help='Depth of FFN')

    parser.add_argument('--seg_thr', type=float, default=0.6,
                        help='Input size')

    parser.add_argument('--mov_thr', type=float, default=0.7,
                        help='Movable threshold')

    parser.add_argument('--act_thr', type=float, default=0.8,
                        help='Activation of segment')

    parser.add_argument('--flex', type=int, default=2,
                        help='Flexibility of the movement policy')

    parser.add_argument('--re_seg_thr', type=int, default=10,
                        help='Threshold for re-segmentation')

    parser.add_argument('--vox_thr', type=int, default=10,
                        help='Voxel threshold - remove if too small')

    parser.add_argument('--manual_seed', type=bool, default=False,
                        help='Specify the seeds source')

    return parser.parse_args()


def canvas_init(args: argparse, process_id: int) -> Tuple[inf.Canvas, Dict]:
    """
    Initialize the FFN model and the Canvas.
    """
    images = tools.load_raw_image(args.data)
    re_seged_count_mask = tools.resume_re_segd_count_mask(args.data_save,
                                                          images.shape[:-1])
    model = FFN(in_channels=4, out_channels=1, input_size=args.input_size,
                delta=args.delta, depth=args.depth).cuda()

    assert os.path.isfile(args.model)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    canvas = inf.Canvas(model, images, args.input_size, args.delta,
                        args.seg_thr, args.mov_thr, args.act_thr, args.flex,
                        args.re_seg_thr, args.vox_thr, args.data_save,
                        re_seged_count_mask, args.save_chunk,
                        args.resume_seed, args.manual_seed, process_id)

    seed_dict = {}
    if os.path.exists(args.seed):
        seed_dict = tools.seeds_to_dict(args.seed)

    return canvas, seed_dict


def inf_run(args: argparse, canvas: inf.Canvas, inf_seed: Dict,
            process_id: int, process_num: int) -> None:
    """
    Run the inference thread for all seeds.
    """
    resume = tools.resume_dict_load(args.data_save, 'resume_seed',
                                    args.resume_seed)
    for seed_id in inf_seed.keys():
        if not seed_id % process_num == process_id:
            continue
        if seed_id < args.resume_seed:
            continue
        coord = inf_seed[seed_id]
        coord = tuple(coord)

        # Seed location limiter
        if coord[1] > 130:
            continue
        if canvas.segment_at(coord, seed_id, args.tag):
            if seed_id % 100 == 0:
                resume['resume_seed'] = seed_id
                tools.pickle_obj(resume, 'resume_seed', args.data_save)
            continue


def main(args: argparse) -> None:
    """
    Multiprocess the inference run.
    """
    threads = args.threads

    for thread in range(threads):
        canvas_inf, inf_seed_dict = canvas_init(args, thread)

        # Manual seed run
        inf_seed_dict = {1: [80, 65, 63], 2: [80, 51, 46], 3: [75, 43, 23],
                         4: [77, 46, 33], 5: [76, 77, 77],
                         6: [65, 83, 88], 7: [53, 86, 100],
                         8: [26, 87, 105], 9: [40, 87, 100], 10: [78, 49, 39]}

        # Main process
        canvas_thread = threading.Thread(target=inf_run,
                                         args=(args, canvas_inf, inf_seed_dict,
                                               thread, threads))
        canvas_thread.start()


if __name__ == '__main__':

    arguments = parse_args()
    main(arguments)
