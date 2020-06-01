import h5py
import numpy as np
from scipy import stats
import argparse

parser = argparse.ArgumentParser(description='script to evaluate id recall')
parser.add_argument('--gt', type=str, default='./data_raw3_focus_500_filter1.5.h5', help='ground truth data')
parser.add_argument('--pred', type=str, default='./recon_raw3.h5', help='prediction h5 file')

args = parser.parse_args()


def run():
    with h5py.File(args.gt, 'r') as f:
        label = f['/label'][()]

    id_L, countL = np.unique(label, return_counts=True)
    print(id_L)

    with h5py.File(args.pred, 'r') as f:
        pred = f['/raw'][()]

    merger = 0
    non = 0
    recon_size = 0

    for id in id_L:
        if id == 0:
            continue
        print("********************")
        print("ID in GT", id)
        maskid_L = (label == id)
        maskid_L_size = np.sum(maskid_L)
        print("ID in GT_vol", np.sum(maskid_L))

        idInPred = stats.mode(pred[maskid_L])
        idInPred_id = idInPred[0][0]
        print("idInPred", idInPred_id)

        maskcapnum = (pred[maskid_L] == idInPred_id)

        print("idInPred vol", np.sum(maskcapnum))
        print("recon_percentage", np.sum(maskcapnum) / maskid_L_size)

        maskid_P = (pred == idInPred_id)
        maskid_Psize = np.sum(maskid_P)

        if idInPred[0][0] == 0:
            print("non")
            non += 1
        elif maskid_Psize >= (maskid_L_size * 1.5):
            print("merger")
            merger += 1
        else:
            if np.sum(maskcapnum) / maskid_L_size >= 0.7:
                recon_size += maskid_L_size

    print("non", non)
    print("merger:", merger)
    print("con", recon_size )
    masknum = (label != 0)
    print(np.sum(masknum))
    print("pixel prescision",recon_size / np.sum(masknum))



if __name__ == "__main__":
    run()
