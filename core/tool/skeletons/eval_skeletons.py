import h5py
import numpy as np
from scipy import stats
import argparse
import skimage
parser = argparse.ArgumentParser(description='script to evaluate id recall')
#parser.add_argument('--gtsk', type=str, default='./skel_sample.h5', help='ground truth data')
#parser.add_argument('--pred', type=str, default='./recon_sample.h5', help='prediction h5 file')

args = parser.parse_args()


def run():
    label = skimage.io.imread("sk_raw5_s.tif")

    id_L, countL = np.unique(label, return_counts=True)
    print(id_L)

    #with h5py.File(args.pred, 'r') as f:
        #pred = f['/raw'][()]
    pred = skimage.io.imread("raw5_recon_grey_s.tif")

    recon_percentage_total = 0
    precision_total = 0

    object_cnt = 1
    for id in id_L:
        if id == 0:
            continue
        print("********************")
        maskid_L = (label == id)
        maskid_L_size = np.sum(maskid_L)
        print("sk_point", maskid_L_size,"id:",id)

        idInPred = stats.mode(pred[maskid_L])


        #mask_inter = ((pred[maskid_L])!=0)
        #print(np.sum(mask_inter))

        idInPred = stats.mode(pred[maskid_L])
        uni, count = np.unique(pred[maskid_L], return_counts=True)

        idInPred_id = idInPred[0][0]
        if idInPred_id == 0:
            count_sort_ind = np.argsort(-count)
            arr_inter = uni[count_sort_ind]
            try:
                idInPred_id = arr_inter[1]
            except IndexError:
                continue


        print("idInPred", idInPred_id)


        maskcapnum = (pred[maskid_L] == idInPred_id)

        print("idInPred vol", np.sum(maskcapnum))
        print("recon_percentage", np.sum(maskcapnum) / maskid_L_size)
        recon_percentage_total += np.sum(maskcapnum) / maskid_L_size
        print(recon_percentage_total)

        # prescision
        idInPred_mask_raw = (pred == idInPred_id)
        label_mask = (label > 0)
        idInPred_mask = (idInPred_mask_raw*label_mask)
        precision_total += np.sum(maskcapnum) / np.sum(idInPred_mask)



        object_cnt += 1




    print("recall",recon_percentage_total/object_cnt)
    print("prescision", precision_total/ object_cnt)


if __name__ == "__main__":
    run()
