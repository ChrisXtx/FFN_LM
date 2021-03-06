import argparse
import time
import random
from torch.utils.data import DataLoader
from functools import partial
import os
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from core.models.ffn import FFN
from core.data import BatchCreator
from scipy.special import expit
from scipy.special import logit
from torch.utils.tensorboard import SummaryWriter
from core.tool.tools import *
from core.data.utils import *
import os

parser = argparse.ArgumentParser(description='Train a network.')

parser.add_argument('--deterministic', action='store_true',
                    help='Run in fully deterministic mode (at the cost of execution speed).')
# load_and_save
parser.add_argument('-train_data', '--train_data_dir', type=str,
                    default='/home/x903102883/2017EXBB/train_data_downsample_2_bound24/train_sample/',
                    help='training data')

parser.add_argument('--resume', type=str,
                    default=None,
                    help='resume model')

parser.add_argument('--save_path', type=str,
                    default='/home/x903102883/FFN_LM_v0.2/model/',
                    help='model save path')
parser.add_argument('--save_interval', type=int, default=2000, help='model save interval')
parser.add_argument('--tag', type=str, default="s_card_sgd", help='tag')

# training parameters
parser.add_argument('-b', '--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='training learning rate')
parser.add_argument('--depth', type=int, default=26, help='depth of ffn')
parser.add_argument('--delta', default=(4, 4, 4), help='delta offset')
parser.add_argument('--input_size', default=(39, 39, 39), help='input size')
parser.add_argument('--opt', default='sgd', help='optimizer')

parser.add_argument('--tb', type=str, default=None, help='path_of_tensorboard')
parser.add_argument('--resume_step', type=int, default=0, help='start_step_of_tb')
parser.add_argument('--iter', type=int, default=1e100, help='training iteration')

# standby
parser.add_argument('--clip_grad_thr', type=float, default=0.7, help='grad clip threshold')
parser.add_argument('--interval', type=int, default=120, help='How often to save model (in seconds).')
parser.add_argument('--gamma', type=float, default=0.9, help='multiplicative factor of learning rate decay')
parser.add_argument('--step', type=int, default=1e5 * 5, help='adjust learning rate every step')

args = parser.parse_args()

deterministic = args.deterministic

if deterministic:
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.benchmark = True  # Improves overall performance in *most* cases

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


def run():
    """model init"""
    model = FFN(in_channels=4, out_channels=1, input_size=args.input_size, delta=args.delta, depth=args.depth).cuda()

    """model resume"""
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))

    if os.path.exists(args.save_path + 'resume_step.pkl'):
        resume = load_obj(args.save_path + 'resume_step.pkl')
    else:
        resume = {'resume_step': args.resume_step}
    args.resume_step = resume['resume_step']
    print('resume_step', args.resume_step)

    if args.tb is None:
        tb = SummaryWriter('./tensorboard/' + args.tag + 'tb_train_log_fov:{}_delta:{}_depth:{}.pth'
                           .format(list(args.input_size)[0], list(args.delta)[0], args.depth))
    else:
        tb = SummaryWriter(args.tb)

    sorted_files_train_data = sort_files(args.train_data_dir)
    files_total = len(sorted_files_train_data)
    input_h5data_dict = {}
    train_dataset_dict = {}
    train_loader_dict = {}
    batch_it_dict = {}
    for index in range(files_total):
        input_h5data_dict[index] = [(args.train_data_dir + sorted_files_train_data[index])]
        train_dataset_dict[index] = BatchCreator(input_h5data_dict[index], args.input_size, delta=args.delta,train=True)
        train_loader_dict[index] = DataLoader(train_dataset_dict[index], shuffle=True, num_workers=0, pin_memory=True)
        batch_it_dict[index] = get_batch(train_loader_dict[index], args.batch_size, args.input_size,
                                         partial(fixed_offsets, fov_moves=train_dataset_dict[index].shifts))

    """optimizer"""
    if args.opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
    # optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step, gamma=args.gamma, last_epoch=-1)

    """train_loop"""
    t_last = time.time()
    cnt = 0
    tp = fp = tn = fn = 0
    best_loss = np.inf
    while cnt < args.iter:
        cnt += 1

        # record training iter every 1000
        if cnt % 1000 == 0:
            resume['resume_step'] = cnt + args.resume_step
            pickle_obj(resume, 'resume_step', args.save_path)

        # load training data (random)
        train_num = len(input_h5data_dict)
        index_rand = random.randrange(0, train_num, 1)
        seeds, images, labels, offsets = next(batch_it_dict[index_rand])
        print(input_h5data_dict[index_rand])
        
        
        t_curr = time.time()
        labels = labels.cuda()
        torch_seed = torch.from_numpy(seeds)
        input_data = torch.cat([images, torch_seed], dim=1)
        input_data = Variable(input_data.cuda())

        logits = model(input_data)
        updated = torch_seed.cuda() + logits

        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(updated, labels)
        loss.backward()
        #torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_thr)
        optimizer.step()

        seeds[...] = updated.detach().cpu().numpy()

        pred_mask = (updated >= logit(0.8)).detach().cpu().numpy()
        true_mask = (labels > 0.5).cpu().numpy()
        true_bg = np.logical_not(true_mask)
        pred_bg = np.logical_not(pred_mask)
        tp += (true_mask & pred_mask).sum()
        fp += (true_bg & pred_mask).sum()
        fn += (true_mask & pred_bg).sum()
        tn += (true_bg & pred_bg).sum()
        precision = 1.0 * tp / max(tp + fp, 1)
        recall = 1.0 * tp / max(tp + fn, 1)
        accuracy = 1.0 * (tp + tn) / (tp + tn + fp + fn)
        print('[Iter_{}:, loss: {:.4}, Precision: {:.2f}%, Recall: {:.2f}%, Accuracy: {:.2f}%]\r'.format(
            cnt, loss.item(), precision * 100, recall * 100, accuracy * 100))

        # scheduler.step()

        """model_saving_(iter)"""

        if (cnt % args.save_interval) == 0:
            tp = fp = tn = fn = 0
            # t_last = t_curr
            # best_loss = loss.item()
            input_size_r = list(args.input_size)
            delta_r = list(args.delta)
            torch.save(model.state_dict(), os.path.join(args.save_path, (
                    str(args.tag) + 'ffn_model_fov:{}_delta:{}_depth:{}.pth'.format(input_size_r[0],
                                                                                    delta_r[0],
                                                                                    args.depth))))
            torch.save(model.state_dict(), os.path.join(args.save_path, (
                    str(args.tag) + 'ffn_model_fov:{}_delta:{}_depth:{}_pre{}_recall{}_.pth'.format(input_size_r[0],
                                                                                              delta_r[0],
                                                                                              args.depth, precision * 100,
                                                                                              recall * 100))))

            print('Precision: {:.2f}%, Recall: {:.2f}%, Accuracy: {:.2f}%, Model saved!'.format(
                precision * 100, recall * 100, accuracy * 100))

            buffer_step = 3000
            resume_step = args.resume_step - buffer_step
            if cnt > buffer_step:
                tb.add_scalar("Loss", loss.item(), cnt + resume_step)
                tb.add_scalar("Precision", precision * 100, cnt + resume_step)
                tb.add_scalar("Recall", recall * 100, cnt + resume_step)
                tb.add_scalar("Accuracy", accuracy * 100, cnt + resume_step)


if __name__ == "__main__":
    

    run()



"""model_saving_(best_loss)"""
"""
        if best_loss > loss.item() or t_curr - t_last > args.interval:
            tp = fp = tn = fn = 0
            t_last = t_curr
            best_loss = loss.item()
            input_size_r = list(args.input_size)
            delta_r = list(args.delta)
            torch.save(model.state_dict(), os.path.join(args.save_path,
                                                        'ffn_model_fov:{}_delta:{}_depth:{}.pth'.format(input_size_r[0],
                                                                                                        delta_r[0],
                                                                                                        args.depth)))
            print('Precision: {:.2f}%, Recall: {:.2f}%, Accuracy: {:.2f}%, Model saved!'.format(
                precision * 100, recall * 100, accuracy * 100))
"""
