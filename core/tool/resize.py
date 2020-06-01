
import cv2
import numpy as np
import argparse
from collections import defaultdict
import h5py
import numpy as np
import tifffile
import skimage
import scipy.ndimage

img = tifffile.TiffFile('data_large.tif').asarray()

print(img.shape)


res = scipy.ndimage.zoom(img,(2, 1, 1 ,1))
#res = cv2.resize(img, dsize=(500, 250, 250, 3), interpolation=cv2.INTER_CUBIC)


#mask = self.seed[tuple(sel_i)] >= self.seg_thr
#self.seg_prob_i[tuple(sel_i)][mask] = quantize_probability(expit(self.seed[tuple(sel_i)][mask]))

skimage.io.imsave('data_large_exp.tif', res)