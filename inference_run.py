import os
import h5py
import argparse
from core.models.ffn import FFN
from core.data.utils import *

parser = argparse.ArgumentParser(description='inference script')
parser.add_argument('--data', type=str, default='./data.h5', help='input images')
parser.add_argument('--label', type=str, default='./pred.h5', help='input images')
parser.add_argument('--model', type=str, default='./model/raw_3_with_color_15_15_15_3_3_3_final.pth', help='path to ffn model')
parser.add_argument('--delta', default=(3, 3, 3), help='delta offset')
parser.add_argument('--input_size', default=(31, 31, 31), help='input size')
parser.add_argument('--depth', type=int, default=12, help='depth of ffn')
parser.add_argument('--seg_thr', type=float, default=0.6, help='input size')
parser.add_argument('--mov_thr', type=float, default=0.9, help='input size')
parser.add_argument('--act_thr', type=float, default=0.95, help='input size')

args = parser.parse_args()


def run():
    """创建模型"""
    model = FFN(in_channels=4, out_channels=1, input_size=args.input_size, delta=args.delta, depth=args.depth).cuda()

    assert os.path.isfile(args.model)

    """载入模型"""
    model.load_state_dict(torch.load(args.model))

    """读取数据"""
    with h5py.File(args.data, 'r') as f:
        images = (f['image'][()].astype(np.float32) - 128) / 33
        # labels = g['label'].value

    """创建分割实例"""
    canva = Canvas(model, images, args.input_size, args.delta, args.seg_thr, args.mov_thr, args.act_thr)
    """开始分割"""
    canva.segment_all()
    """获取结果"""
    # result = canva.segmentation
    # """存储结果"""
    # max_value = result.max()
    # indice, count = np.unique(result, return_counts=True)
    # result[result == -1] = 0
    # result = result*(1.0 * 255/max_value)
    rlt_key = []
    rlt_val = []
    result = canva.target_dic
    with h5py.File(args.label, 'w') as g:
        for key, value in result.items():
            rlt_key.append(key)
            rlt_val.append((value > 0).sum())
            g.create_dataset('id_{}'.format(key), data=value.astype(np.uint8), compression='gzip')
    print('label: {}, number: {}'.format(rlt_key, rlt_val))


if __name__ == '__main__':
    run()
