"""Runs inference on both manually provided and automatically generated seeds.

Inference is multi-processed.
"""
import os
import argparse
from typing import Dict
from core.models.ffn import FFN
import core.tool.tools as tools
import threading
import torch
from core.inference.inference import Canvas


def parse_args() -> argparse:
    """Parse the inference arguments."""
    # TODO: too many arguments here, move most to a settings/config file.
    parser = argparse.ArgumentParser(description='Inference Arguments Parser')

    parser.add_argument('--data', type=str, required=True,
                        help='Path to input images')

    parser.add_argument('--seed', type=str, required=True,
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


def load_model(args: argparse) -> FFN:
    """Load saved FFN model."""
    model = FFN(in_channels=4, out_channels=1, input_size=args.input_size,
                delta=args.delta, depth=args.depth).cuda()

    assert os.path.isfile(args.model)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    return model


def canvas_init(args: argparse, process_id: int) -> Canvas:
    """Initialize the Canvas."""
    images = tools.load_raw_image(args.data)
    re_seged_count_mask = tools.resume_re_segd_count_mask(args.data_save,
                                                          images.shape[:-1])

    model = load_model(args)

    canvas = Canvas(model, images, args.input_size, args.delta,
                    args.seg_thr, args.mov_thr, args.act_thr, args.flex,
                    args.re_seg_thr, args.vox_thr, args.data_save,
                    re_seged_count_mask, args.save_chunk,
                    args.resume_seed, args.manual_seed, process_id)

    return canvas


def run(args: argparse, canvas: Canvas, seeds: Dict, process_id: int,
        process_num: int) -> None:
    """Run the inference thread for all seeds."""
    resume = tools.resume_dict_load(args.data_save, 'resume_seed',
                                    args.resume_seed)
    
    seeds_part = int(len(inf_seed_dict)/process_num)
    for seed_id, coord in seeds.items():
        if (seed_id < (seeds_part*(process_id-1))) | (seed_id > (seeds_part*process_id)):
            continue
        if seed_id < args.resume_seed:
            continue
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
    """Multiprocess the inference run."""
    threads = args.threads

    for thread in range(threads):
        canvas = canvas_init(args, thread)
        seeds = tools.load_seeds_from_file(args.seed, args.manual_seed)

        # Main process
        canvas_thread = threading.Thread(target=run,
                                         args=(args, canvas, seeds, thread,
                                               threads))
        canvas_thread.start()


if __name__ == '__main__':

    arguments = parse_args()
    main(arguments)
