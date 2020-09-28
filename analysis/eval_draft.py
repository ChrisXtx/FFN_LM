import h5py
import numpy as np
from scipy import stats
import argparse
import skimage.io
from collections import Counter
from pathlib import Path
import natsort
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description='script to evaluate id recall')

parser.add_argument('--gt', type=str, default='', help='ground truth data')
parser.add_argument('--pred', type=str, default='', help='folder of predictions data')
parser.add_argument('--size_thr', type=int, default=500, help='ground truth data')


args = parser.parse_args()


def sort_files(dir_path):
    entries = Path(dir_path)
    files = []
    for entry in entries.iterdir():
        files.append(entry.name)
    sorted_files = natsort.natsorted(files, reverse=False)
    return sorted_files


def evaluation(label, pred, file_name):
    #returns the stats of evluation 
    ids_l, count_l = np.unique(label, return_counts=True)
    fail_r_p = fail_cnt = fail_r = fail_size = fail_p = \
    t_recall = t_precision = t_f1score = t_obj = small_cnt = 0
    recall_list = []
    precision_list = []
    f1score_list = []

    for id_l in ids_l:

        gt_id = (label == id_l)
        gt_id_size = np.sum(gt_id)

        # size filter for objects in gt
        if gt_id_size < args.size_thr:
            small_cnt += 1
            continue

        # finding the candidate object in the prediction
        id_stat_pred = stats.mode(pred[gt_id])
        target = id_stat_pred[0][0]

        # excluding zero (background)
        if id_stat_pred[0][0] == 0:
            ctr = Counter(pred[gt_id].ravel())
            try:
                second_id, frequency = ctr.most_common(2)[1]
                target = second_id
            except IndexError:
                continue

        tp = (pred[gt_id] == target)
        positive = (pred == target)
        positive_size = np.sum(positive)
        precision = np.sum(tp) / positive_size
        recall = np.sum(tp) / gt_id_size
        f1score = 2 * (precision * recall) / (precision + recall)

        if (np.sum(tp) < 400) | (np.sum(tp) < (gt_id_size * 0.25)) \
            | (np.sum(tp) < (positive_size * 0.25)):
            fail_cnt += 1
            f1score = recall = precision = 0
        if np.sum(tp) < 400:
            fail_size += 1
        if np.sum(tp) < (gt_id_size*0.25):
            fail_r += 1 
            fail_r_p += 1
        if np.sum(tp) < positive_size*0.25:
            fail_p += 1
            fail_r_p += 1

        if f1score != 0:
            t_f1score += f1score
            t_recall += recall
            t_precision += precision
            t_obj += 1

        recall_list.append(recall)
        precision_list.append(precision)
        f1score_list.append(f1score)

    print(file_name + " avg recall: {:.2f}%  avg precision: {:.2f}%  avg F1: {:.2f}%  fail rate: {:.2f}%  "
          .format(100*t_recall / t_obj, 100*t_precision / t_obj, 100*t_f1score / t_obj, 100*fail_cnt/(t_obj+fail_cnt)))
    print("objects in gt: {}  too small: {} reconstructed: {}".format(len(ids_l), small_cnt, t_obj))
    print("fail_total: {} fail_Size: {}  fail_Recall: {}  fail_Precision: {}  fail_R&P: {}"
          .format(fail_cnt, fail_size, fail_r, fail_p, fail_r_p))

    return recall_list, precision_list, f1score_list
