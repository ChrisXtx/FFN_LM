import os
import h5py
import argparse
from core.models.ffn import FFN
from core.data.utils import *
from core.tool.tools import *
from core.tool.SWC.swc_parser import *

parser = argparse.ArgumentParser(description='inference script')
parser.add_argument('--data', type=str, default='./raw.tif', help='input images')
parser.add_argument('--model', type=str, default='./model.pth', help='path to ffn model')
parser.add_argument('--seeds', type=str, default='./seeds.h5', help='path to seeds')

parser.add_argument('--data_save', type=str, default='./data_save',help='swc_skeletons')
parser.add_argument('--save_chunk', type=int, default=5000, help='separate the seg_coords from seeds by chunk')
parser.add_argument('--delta', default=(4, 4, 4), help='delta offset')
parser.add_argument('--input_size', default=(39, 39, 39), help='input size')
parser.add_argument('--depth', type=int, default=26, help='depth of ffn')
parser.add_argument('--seg_thr', type=float, default=0.6, help='input size')
parser.add_argument('--mov_thr', type=float, default=0.6, help='movable thr')
parser.add_argument('--act_thr', type=float, default=0.8, help='activation of seg')
parser.add_argument('--vox_thr', type=int, default=500, help='remove if too small')


args = parser.parse_args()


def run():

    # model init
    model = FFN(in_channels=4, out_channels=1, input_size=args.input_size, delta=args.delta, depth=args.depth).cuda()
    assert os.path.isfile(args.model)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    # load raw image
    if args.data[-2:] == 'h5':
        with h5py.File(args.data, 'r') as f:
            images = (f['/image'][()].astype(np.float32) - 128) / 33
    else:
        images = ((skimage.io.imread(args.data)).astype(np.float32) - 128) / 33

    # inference construction
    canvas_inf = Canvas(model, images, args.input_size, args.delta, args.seg_thr, args.mov_thr,
                        args.act_thr, args.vox_thr,  args.data_save, args.save_chunk)

    inf_seed_dict = {}
    with h5py.File(args.seed, 'r') as segs:

        seeds = segs['seeds'][()]
        seed_id = 1
        for coord in seeds:
            inf_seed_dict[seed_id] = coord


    # run inference on every seed and save their segmentation
    for seed_id in inf_seed_dict.keys():

        coord = inf_seed_dict[seed_id]
        if canvas_inf.segment_at(coord, seed_id):
            continue


if __name__ == '__main__':
    run()
