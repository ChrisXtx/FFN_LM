
import h5py
import numpy as np
import skimage
import cv2
import skimage.feature
import random
from scipy import ndimage

#with h5py.File('/home/x903102883/FFN_LM/training_data/train_raw5/raw5_label_final.h5', 'r') as f:
    #labels = f['/label'][()]
labels = skimage.io.imread("/home/x903102883/2017EXBB/16bit/labels_tiff/128/eboyden-1_(768, 2816, 0).tif")


unique_sk = np.unique(labels)
print(unique_sk)


skel = np.zeros(labels.shape)


for id in unique_sk:
    print(id)

    mask_sk = (labels == id)
    label = np.zeros(labels.shape)
    label[mask_sk] = 1


    edges = ndimage.generic_gradient_magnitude(
                label.astype(np.float32),
                ndimage.sobel)
    sigma = 49.0 / 6.0
    thresh_image = np.zeros(edges.shape, dtype=np.float32)
    ndimage.gaussian_filter(edges, sigma, output=thresh_image, mode='reflect')
    filt_edges = edges > thresh_image

    del edges, thresh_image
    dt = ndimage.distance_transform_edt(1 - filt_edges).astype(np.float32)

    state = np.random.get_state()
    np.random.seed(42)
    idxs = skimage.feature.peak_local_max(
    dt + np.random.random(dt.shape) * 1e-4,
    indices=True, min_distance=1, threshold_abs=0, threshold_rel=0)
    np.random.set_state(state)
    for coord in idxs:
        print(coord)

        x = int(coord[2])
        y = int(coord[1])
        z = int(coord[0])
        rad = 1

        skel[z , y , x ] = id
        #for xd in range(-rad, rad + 1):
            #for yd in range(-rad, rad + 1):
                #for zd in range(-rad, rad + 1):
                    #skel[z + zd, y + yd, x + xd] = id




ids = np.unique(skel)
                # print(ids)

stacked_img = np.stack((skel,) * 3, axis=-1)

for id_i in ids:
                    # print(id_i)
    id_mask = (skel == id_i)

    rad2 = random.randrange(10, 254, 1)
    rad3 = random.randrange(10, 254, 1)
    rad1 = random.randrange(10, 254, 1)
    if id_i == 0:
        rad1 = rad2 = rad3 = 0
    stacked_img[id_mask] = (rad1, rad2, rad3)


unique= np.unique(skel)
print(unique)

skimage.io.imsave("sk_raw5.tif",skel.astype('uint8'))



"""
skel = np.zeros(labels.shape)
stacked_img = np.stack((skel,) * 3, axis=-1)

for id_i in unique_sk:
                    # print(id_i)
    id_mask = (labels == id_i)

    rad2 = random.randrange(10, 254, 1)
    rad3 = random.randrange(10, 254, 1)
    rad1 = random.randrange(10, 254, 1)
    if id_i == 0:
        rad1 = rad2 = rad3 = 0
    stacked_img[id_mask] = (rad1, rad2, rad3)


skimage.io.imsave("GT.tif",stacked_img.astype('uint8'))
"""