"""Runs FFN inference."""

import os
import h5py
import argparse
import torch
import numpy as np
from core.models.ffn import FFN
from core.data.utils import Canvas
import skimage
parser = argparse.ArgumentParser(description='Inference script')
parser.add_argument('--data', type=str, default='/home/x903102883/2017EXBB/train_data_sep/sparse/(0, 512, 0)_raw.h5', help='Input images')
parser.add_argument('--model', type=str, default='/home/x903102883/2017EXBB/whole_volume_inf/down_2_adamffn_model_fov_39_delta_4_depth_26_recall87.6557408472302.pth',
                    help='Path to FFN model')
parser.add_argument('--seeds', type=str, default=None,
                    help='Path to seed coordinates file')
parser.add_argument('--output', type=str, default='/home/x903102883/2017EXBB/whole_volume_inf/',
                    help='Path to output files')
parser.add_argument('--delta', default=(15, 15, 15),
                    help='Delta offset')
parser.add_argument('--input_size', default=(51, 51, 51), help='Input size')
parser.add_argument('--depth', type=int, default=26, help='Depth of FFN')
parser.add_argument('--seg_thr', type=float, default=0.6,
                    help='Segment threshold')
parser.add_argument('--mov_thr', type=float, default=0.7,
                    help='Move threshold')
parser.add_argument('--act_thr', type=float, default=0.8,
                    help='Act threshold')
parser.add_argument('--tag', type=str, default='single_seed_inf_', help='tag the files')
args = parser.parse_args()


def run():
    """Run the inference."""
    model = FFN(in_channels=4, out_channels=1, input_size=args.input_size,
                delta=args.delta, depth=args.depth).cuda()

    assert os.path.isfile(args.model)

    if args.model is not None:
        checkpoint_no_module = {}

        pretrained_dict = torch.load(args.model, map_location=lambda storage,
                                     loc: storage)
        for k, v in pretrained_dict.items():
            # If the model was saved with 'DataParallel' it will have a
            # 'module' prefix. As we are loading the model without
            # DataParallel now, remove 'module' preifx if it exits.
            if k.startswith('module'):
                # Remove 'module' prefix from the keys
                k = k[7:]
            checkpoint_no_module[k] = v
        info = model.load_state_dict(checkpoint_no_module, strict=False)
        print(info)

    model.eval()

    # Open the input image file and read the stack of images
    if args.data[-2:] == 'h5':
        with h5py.File(args.data, 'r') as f:
            print(f.keys())
            images = (f['/images'][()].astype(np.float32) - 128) / 33
    else:
        images = ((skimage.io.imread(args.data)).astype(np.float32) - 128) / 33

    seed_list = []
    if args.seeds is not None:
        # Open the seed coordinate file and read in the seeds
        with h5py.File(args.seeds, 'r') as f:
            seed_list = f['/inference_coor'][()].astype(np.int64)

    canva = Canvas(model, args.input_size, args.delta,
                   args.seg_thr, args.mov_thr, args.act_thr, re_seg_thr=5,vox_thr=100,data_save_path=args.output,
                   process_id=1)

    # TODO: Modify segment_all to process seed_list
    # For now use segment_at instead
    if len(seed_list) > 0:
        # Segment for each seed position
        for seed in seed_list:
            canva.segment_at((seed[0], seed[1], seed[2]), 0)
    else:
        # Manually specify a seed location
        # TODO: input seed location from user
        canva.segment_at((42, 184, 40), 0, args.tag)

    # canva.segment_all()

    # result = canva.segmentation
    # max_value = result.max()
    # indice, count = np.unique(result, return_counts=True)
    # result[result == -1] = 0
    # result = result*(1.0 * 255/max_value)

    # rlt_key = []
    # rlt_val = []
    # result = canva.target_dic
    # with h5py.File(args.label, 'w') as g:
    #     for key, value in result.items():
    #         rlt_key.append(key)
    #         rlt_val.append((value > 0).sum())
    #         g.create_dataset('id_{}'.format(key),
    #      data=value.astype(np.uint8), compression='gzip')
    # print('label: {}, number: {}'.format(rlt_key, rlt_val))


if __name__ == '__main__':
    run()
