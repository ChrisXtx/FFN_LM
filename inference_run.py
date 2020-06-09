"""Runs FFN inference."""

import os
import h5py
import argparse
import torch
import numpy as np
from core.models.ffn import FFN
from core.data.utils import Canvas

parser = argparse.ArgumentParser(description='inference script')
parser.add_argument('--data', type=str, default='', help='input images')
parser.add_argument('--model', type=str, default='',
                    help='path to ffn model')
parser.add_argument('--delta', default=(4, 4, 4),
                    help='delta offset')
parser.add_argument('--input_size', default=(39, 39, 39), help='input size')
parser.add_argument('--depth', type=int, default=12, help='depth of ffn')
parser.add_argument('--seg_thr', type=float, default=0.6, help='input size')
parser.add_argument('--mov_thr', type=float, default=0.7, help='input size')
parser.add_argument('--act_thr', type=float, default=0.8, help='input size')
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

    with h5py.File(args.data, 'r') as f:
        images = (f['/image'][()].astype(np.float32) - 128) / 33

    seed_list = []
    canva = Canvas(model, images, seed_list, args.input_size, args.delta,
                   args.seg_thr, args.mov_thr, args.act_thr)

    canva.segment_at((106, 137, 173), 2)

    # canva.segment_all()

    # result = canva.segmentation
    #
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
