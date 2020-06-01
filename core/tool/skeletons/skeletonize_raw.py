import h5py
import numpy as np
import skimage
import cv2
import skimage.feature
import random
from scipy import ndimage


data = skimage.io.imread("/home/x903102883/FFN_LM_v0.2/core/tool/skeletons/test_data/eboyden-1_(768, 2816, 0)_raw_binary.tif")
print(data.shape)
skel = np.zeros(data.shape)
mask_sk = (data >= 0)
print(np.sum(mask_sk))
bi_mask = np.zeros(data.shape)
bi_mask[mask_sk] = 1
print(np.sum(bi_mask))

edges = ndimage.generic_gradient_magnitude(
                bi_mask.astype(np.float32),
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

print(idxs)
np.random.set_state(state)

for coord in idxs:
    print(coord)

    x = int(coord[2])
    y = int(coord[1])
    z = int(coord[0])
    rad = 1

    skel[z , y , x ] = 200

skimage.io.imsave("./test_sk.tif",skel.astype('uint8'))