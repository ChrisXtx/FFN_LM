import os
import h5py
import argparse
from core.models.ffn import FFN
from core.data.utils import *
from core.tool.tools import *
from core.tool.SWC.swc_parser import *

parser = argparse.ArgumentParser(description='inference script')
parser.add_argument('--data', type=str,
                    default='/home/x903102883/FFN_LM_v0.2/data/agglomeration_net_test/pf_axonal_1/pf_raw_his_axonal_1_dim_down2.tif',
                    help='input images')
parser.add_argument('--swc', type=str,
                    default='/home/x903102883/2017EXBB/PF_inf/pf/axnol/axnol_fused.swc',
                    help='swc_skeletons')
parser.add_argument('--model', type=str,
                    default='/home/x903102883/FFN_LM_v0.2/model/down_2_adamffn_model_fov:39_delta:4_depth:26_recall86.85200922666019.pth',
                    help='path to ffn model')

parser.add_argument('--data_save', type=str,
                    default='/home/x903102883/FFN_LM_v0.2/data/agglomeration_net_test/pf_axonal_1/',
                    help='swc_skeletons')

parser.add_argument('--save_chunk', type=int, default=5000, help='separate the seg_coords from seeds by chunk')
parser.add_argument('--buffer_distance', type=int, default=0, help='swc_skeletons_dis_from_overlap')
parser.add_argument('--delta', default=(4, 4, 4), help='delta offset')
parser.add_argument('--input_size', default=(39, 39, 39), help='input size')
parser.add_argument('--swc_scaling', type=int, default=2, help='swc_scaling')
parser.add_argument('--depth', type=int, default=26, help='depth of ffn')
parser.add_argument('--seg_thr', type=float, default=0.6, help='input size')
parser.add_argument('--mov_thr', type=float, default=0.8, help='movable thr')
parser.add_argument('--act_thr', type=float, default=0.8, help='activation of seg')
parser.add_argument('--re_seg_thr', type=int, default=3, help='will not seed here if segmented many times')
parser.add_argument('--vox_thr', type=int, default=500, help='remove if too small')
parser.add_argument('--resume_seed', type=int, default=0, help='resume_seed')

args = parser.parse_args()


def run():
    model = FFN(in_channels=4, out_channels=1, input_size=args.input_size, delta=args.delta, depth=args.depth).cuda()
    assert os.path.isfile(args.model)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    if args.data[-2:] == 'h5':
        with h5py.File(args.data, 'r') as f:
            images = (f['/image'][()].astype(np.float32) - 128) / 33
    else:
        images = ((skimage.io.imread(args.data)).astype(np.float32) - 128) / 33

    inf_seed_display = np.zeros(images.shape[:-1])
    inf_seed_dict, inf_seed_list = swc_to_inf_seeds(args.swc, images.shape, args.swc_scaling,
                                                    args.input_size, args.delta, args.buffer_distance, inf_seed_display,
                                                    args.data_save)

    with h5py.File(args.data[:-4] + "inf_record.h5", 'w') as f:
        f.create_dataset('image_raw', data=images, compression='gzip')
        f.create_dataset('inf_coords', data=inf_seed_list, compression='gzip')
        f.create_dataset('inf_seed_display', data=inf_seed_display, compression='gzip')

    canvas_inf = Canvas(model, images, args.input_size, args.delta, args.seg_thr, args.mov_thr,
                        args.act_thr, args.re_seg_thr, args.vox_thr, args.data_save, args.save_chunk,args.resume_seed)


    # run inference on every seed and save their segmentation

    
    inf_cnt = 0
    for seed_id in inf_seed_dict.keys():
        inf_cnt += 1
        if inf_cnt < args.resume_seed:
          continue
        coord = inf_seed_dict[seed_id]

        if canvas_inf.segment_at(coord, seed_id):
            continue


if __name__ == '__main__':
    run()
