import os
import h5py
import argparse
from core.models.ffn import FFN
from core.data.utils import *
from collections import OrderedDict

parser = argparse.ArgumentParser(description='inference script')
parser.add_argument('--data', type=str, default='./input/roi/roi-d2x-873-2850.h5', help='input images')
parser.add_argument('--model', type=str, default='./model/downsample_2_ffn_model_fov_39_delta_4_depth_12.pth', help='path to ffn model')
parser.add_argument('--delta', default=(4, 4, 4), help='delta offset')
parser.add_argument('--input_size', default=(39, 39, 39), help='input size')
parser.add_argument('--depth', type=int, default=12, help='depth of ffn')
parser.add_argument('--seg_thr', type=float, default=0.5, help='input size')
parser.add_argument('--mov_thr', type=float, default=0.7, help='input size')
parser.add_argument('--act_thr', type=float, default=0.8, help='input size')

args = parser.parse_args()


def run():
    
    model = FFN(in_channels=4, out_channels=1, input_size=args.input_size, delta=args.delta, depth=args.depth).cuda()

    assert os.path.isfile(args.model)
     
    # The model was saved with 'DataParallel' that stores the model with 
    # 'module'. Now we are trying to load the model without DataParallel so
    # load the weights file, create a new ordered dict without the module prefix.
 
    # Load saved model file
    state_dict = torch.load(args.model)
   
    # Create a new OrderedDict that does not contain `module.`
    state_dict_without_module = OrderedDict()
    for k, v in state_dict.items():
         # Remove 'module' prefix from the keys
         name = k[7:]
         state_dict_without_module[name] = v
    
    # Load params
    model.load_state_dict(state_dict_without_module)

    model.eval()

    
    with h5py.File(args.data, 'r') as f:
        images = (f['/image'][()].astype(np.float32) - 128) / 33

    seed_list = []
    canva = Canvas(model, images, seed_list, args.input_size, args.delta, args.seg_thr, args.mov_thr, args.act_thr)
    canva.segment_at((42, 92, 20), 0)


    #canva.segment_all()
    
    # result = canva.segmentation
    # 
    # max_value = result.max()
    # indice, count = np.unique(result, return_counts=True)
    # result[result == -1] = 0
    # result = result*(1.0 * 255/max_value)

    """
    rlt_key = []
    rlt_val = []
    result = canva.target_dic
    with h5py.File(args.label, 'w') as g:
        for key, value in result.items():
            rlt_key.append(key)
            rlt_val.append((value > 0).sum())
            g.create_dataset('id_{}'.format(key), data=value.astype(np.uint8), compression='gzip')
    print('label: {}, number: {}'.format(rlt_key, rlt_val))
    """


if __name__ == '__main__':
    run()
