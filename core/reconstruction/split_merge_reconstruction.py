
import os
import h5py
import argparse
from core.data.utils import *
import pickle
from core.tool.tools import *
import sys
sys.setrecursionlimit(10**7)


def coor_to_code(array):

    # convert the coordinate [x , y, z] to a unique code
    codes_array = np.zeros((len(array),), dtype=np.int32)
    for index in range(len(array)):
        codes_array[index] = (int(str(array[index][0]) + str(array[index][1]) + str(array[index][2])) +
                              array[index][0] * 1 + array[index][1] * 2 + array[index][2] * 3)
    return codes_array


def span_overlap_check(span1, span2):
    # check if two span volume cubes overlap
    for dim in range(len(span1[0])):
        if (span2[0][dim] >= span1[1][dim]) | (span2[1][dim] <= span1[0][dim]):
            return False
    return True


def convert_coor_all(seg_s_path):
    """
    convert coordinates of each id in .h5 to codes
    :param seg_s_path: h5 file of segmentation (coordinates)
    :return: dict of {id: codes}
    """

    seg_s_files = sort_files(seg_s_path)
    code_dict = {}
    for seg_s_file in seg_s_files:
        with h5py.File(seg_s_path + seg_s_file, 'r') as seg_s:
            ids = list(seg_s.keys())

            for idx in ids:
                coors = seg_s[idx][()]
                codes = coor_to_code(coors)
                code_dict[idx] = codes

    return code_dict


def h5_to_dict(dir, action):

    dict_r = {}
    files = sort_files(dir)
    for file in files:
        if 'part' not in file:
            continue
        with h5py.File(dir + file, 'r') as f:
            ids = list(f.keys())
            for idx in ids:
                data = f[str(idx)][()]
                data_r = action(data)
                dict_r[idx] = data_r
    return dict_r


def merge(seg_s_path, seg_s_code_dict, ratio_thr, vox_thr):
    """ merge segmentation from different seeds if they have enough overlaps

    :param seg_s_path: dir of segmentation
    :param seg_s_code_dict:  {id : codes of coordinates}

    :return: merge_dict {id : set( merged ids ) }
    """

    def get_span(data):
        return [np.amin(data, axis=0), np.amax(data, axis=0)]
    span_dict = h5_to_dict(seg_s_path, get_span)

    merge_dict = {}
    for idx in seg_s_code_dict.keys():
        print("merge_check", idx)
        seg_codes = seg_s_code_dict[idx]
        seg_span = span_dict[idx]

        id_seg_vox = len(seg_codes)
        # TODO: init the set for merge groups
        merge_dict[int(float(idx))] = set()

        for id_merge_exam in seg_s_code_dict.keys():

            # check if it was possible for them to have overlaps
            id_merge_exam_span = span_dict[id_merge_exam]
            if not span_overlap_check(seg_span, id_merge_exam_span):
                continue

            seg_codes_merge_exam = seg_s_code_dict[id_merge_exam]
            id_merge_exam_seg_vox = len(seg_codes_merge_exam)

            # get the number of overlapping codes
            overlap_vox = len(np.intersect1d(seg_codes_merge_exam, seg_codes))

            ratio_self = overlap_vox / id_seg_vox  # overlap / self_segmentation
            ratio_merge_exam = overlap_vox / id_merge_exam_seg_vox  # overlap / merge_exam_target_segmentation

            if (ratio_self <= ratio_thr) | (ratio_merge_exam <= ratio_thr):
                continue
            if overlap_vox < vox_thr:
                continue

            merge_dict[int(float(idx))].add(int(float(id_merge_exam)))

    return merge_dict


def pickle_obj(obj, name, path):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def recur_merge_link(idx, merge_dict, merge_group, done_list):

    # recursively link all the merged objects
    if idx in done_list:
        return
    ids_merge = merge_dict[idx]
    if idx not in merge_group:
        merge_group.append(idx)
    done_list.append(idx)
    for id_merge in ids_merge:
        if idx == id_merge:
            continue
        if id_merge in done_list:
            continue
        if id_merge not in merge_group:
            merge_group.append(id_merge)
        recur_merge_link(id_merge, merge_dict, merge_group, done_list)
    return merge_group


def merge_seg_s(merge_dict):

    # adding all merged segmentation into the same group
    merge_group_dict = {}
    done_list = []
    obj = 0
    for id in merge_dict.keys():
        if id in done_list:
            continue
        merge_group = []
        merge_group = recur_merge_link(id, merge_dict, merge_group, done_list)
        obj += 1
        merge_group_dict[obj] = merge_group

    return merge_group_dict


def coors_to_mask(coors, image_size):
    mask = np.zeros(image_size, dtype=bool)
    z_coors = [c[0] for c in coors]
    y_coors = [c[1] for c in coors]
    x_coors = [c[2] for c in coors]
    mask[z_coors, y_coors, x_coors] = True

    return mask


def seg_s_to_RGB(segmentation):

    """convert a segmentation in unique id space to the RGB(8bit) hue space

    :param segmentation: segmentation with unique id representation
    :return: RGB segmentation
    """
    ids = np.unique(segmentation)
    layer = np.zeros(segmentation.shape).astype('uint8')
    RGB_img = np.stack((layer,) * 3, axis=-1)

    for idx in ids:

        id_mask = (segmentation == idx)
        rand2 = random.randrange(0, 255, 1)
        rand3 = random.randrange(0, 255, 1)
        rand1 = random.randrange(0, 255, 1)
        if idx == 0:
            rand1 = rand2 = rand3 = 0
        RGB_img[id_mask] = (rand1, rand2, rand3)

    return RGB_img


def seg_s_reconstructor(seg_s_path, merge_group_dict, image_shape, cons_thr=1):

    """ consensus / split / agglomeration
    object groups were defined by  merge_group_dict
    id: the seed that started the inference

    :param seg_s_path: (dir)  segmentation data (coordinates) of all seeds
    :param merge_group_dict:
    :param image_shape:
    :param cons_thr:  the times that a coordinate should be included to reach a consensus
    :return:
    """

    segmentation = np.zeros(image_shape, dtype='uint32')
    seg_s_dict = h5_to_dict(seg_s_path, action=lambda data: data)

    num_group = len(merge_group_dict)
    for obj in merge_group_dict.keys():
        print("reonstruct", obj)
        consensus = np.zeros(image_shape, dtype='uint8')

        for idx in merge_group_dict[obj]:
            coors = seg_s_dict[str(idx)]
            id_seg_mask = coors_to_mask(coors, image_shape)
            segmentation[id_seg_mask] = obj
            consensus[id_seg_mask] += 1
        
        # over segmentation split
        if cons_thr > 2:
            consensus_fail = (consensus < cons_thr)
            # clear the  consensus failed region
            segmentation[consensus_fail] = 0
            # over segmentation
            for idx in merge_group_dict[obj]:
                coors = seg_s_dict[str(idx)]
                id_seg_mask = coors_to_mask(coors, image_shape)
                split_mask = (consensus_fail * id_seg_mask)
                segmentation[split_mask] = num_group + idx
        
    return segmentation


