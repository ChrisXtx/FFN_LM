
import os
import h5py
import argparse

from core.data.utils import *
import pickle
from core.tool.tools import *

def coor_to_code(array):
    codes_array = np.zeros((len(array),), dtype=np.int32)
    for index in range(len(array)):
        # need to optimize encoding strategy
        codes_array[index] = (int(str(array[index][0]) + str(array[index][1]) + str(array[index][2])) +
                              array[index][0]*1 + array[index][1]*2 + array[index][2]*3)
    return codes_array


def span_overlap_check(span1, span2):
    # check if two span volume cubes have overlaps
    for dim in range(len(span1[0])):
        if (span2[0][dim] >= span1[1][dim]) | (span2[1][dim] <= span1[0][dim]):
            return False
    return True


def convert_coor_all(segs_path):
    """
    convert coors of each id in .h5 to codes
    :param segs_path: h5 file of coors
    :return:
    """

    segs_files = sort_files(segs_path)
    code_dict = {}
    for segs_file in segs_files:
        with h5py.File(segs_path + segs_file, 'r') as segs:
            ids = list(segs.keys())

            for id in ids:
                coors = segs[id][()]
                codes = coor_to_code(coors)
                code_dict[id] = codes

    return code_dict


def merge(segs_path, segs_code_dict, ratio_thr):
    """
    merge two segs from two seeds if they have enough overlaps
    :param codes_path:
    :param save_path:
    :return: merge_dict {id : set( merge_id ) }
    """


    span_dict = {}
    segs_files = sort_files(segs_path)

    for segs_file in segs_files:
        if 'part' not in segs_file:
            continue
        with h5py.File(segs_path + segs_file, 'r') as segs:
            ids = list(segs.keys())
            for id in ids:
                seg = segs[str(id)][()]
                seg_span = [np.amin(seg, axis=0), np.amax(seg, axis=0)]
                span_dict[id] = seg_span


    merge_dict = {}
    for id in segs_code_dict.keys():
        print("merge_check", id)
        seg_codes = segs_code_dict[id]
        seg_span = span_dict[id]

        id_seg_voxels = len(seg_codes)

        merge_dict[int(float(id))] = set()
        for id_merge_exam in segs_code_dict.keys():

            # check if it was possible for them to overlap
            id_merge_exam_span = span_dict[id_merge_exam]
            if not span_overlap_check(seg_span, id_merge_exam_span):
                continue

            seg_codes_merge_exam = segs_code_dict[id_merge_exam]
            id__merge_exam_seg_voxels = len(seg_codes_merge_exam)

            overlap_voxels = len(np.intersect1d(seg_codes_merge_exam, seg_codes))

            ratio_self = overlap_voxels / id_seg_voxels                    # overlap / self_segmentation
            ratio_merge_exam = overlap_voxels / id__merge_exam_seg_voxels  # overlap / merge_exam_target_segmentation

            if (ratio_self <= ratio_thr) | (ratio_merge_exam <= ratio_thr):
                continue
            else:
                merge_dict[int(float(id))].add(int(float(id_merge_exam)))


    return  merge_dict


def pickle_obj(obj, name, path ):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def recur_merge_link(id, merge_dict, merge_group, done_list):
    # recursively link all the merges
    if id in done_list:
        return
    ids_merge = merge_dict[id]
    if id not in merge_group:
        merge_group.append(id)
    done_list.append(id)
    for id_merge in ids_merge:
        if id == id_merge:
            continue
        if id_merge in done_list:
            continue
        if id_merge not in merge_group:
            merge_group.append(id_merge)
        recur_merge_link(id_merge, merge_dict, merge_group,done_list)
    return merge_group


def merge_segs(merge_dict):

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


def segs_to_RGB(segmented_mask):
    """
    :param segmented mask with unique id representation
    :return: RGB segmentation
    """
    ids = np.unique(segmented_mask)
    RGB_img = np.stack((segmented_mask,) * 3, axis=-1)

    for id in ids:

        id_mask = (segmented_mask == id)
        rand2 = random.randrange(0, 255, 1)
        rand3 = random.randrange(0, 255, 1)
        rand1 = random.randrange(0, 255, 1)
        if id == 0:
            rand1 = rand2 = rand3 = 0
        RGB_img[id_mask] = (rand1, rand2, rand3)

    return RGB_img


def segs_reconstructor(segs_path, merge_group_dict, image_shape, cons_thr=1):
    """ consensus / split / agglomeration
    object group was defined by merge group data
    id is the id of the seed that started the inference
    :param segs_path:
    :param merge_group_dict:
    :param image_shape:
    :param cons_thr:  the times that a voxel should be included to reach a consensus
    :return:
    """
    segmentation = np.zeros(image_shape, dtype='uint8')

    segs_dict = {}
    segs_files = sort_files(segs_path)

    for segs_file in segs_files:
        if 'part' not in segs_file:
            continue
        with h5py.File(segs_path + segs_file, 'r') as segs:
            ids = list(segs.keys())
            for id in ids:
                coors = segs[str(id)][()]
                segs_dict[id] = coors


    num_group = len(merge_group_dict)
    for obj in merge_group_dict.keys():
        print("reonstruct", obj)
        #consensus = np.zeros(image_shape, dtype='uint8')
        for id in merge_group_dict[obj]:

            coors = segs_dict[str(id)]
            id_seg_mask = coors_to_mask(coors, image_shape)
            segmentation[id_seg_mask] = obj
            #consensus[id_seg_mask] += 1
        """
        # over segmentation split
        if cons_thr > 1:
            consensus_fail = (consensus < cons_thr)
            # clear the  consensus failed region
            segmentation[consensus_fail] = 0
            # over segmentation
            for id in merge_group_dict[obj]:
                coors = segs_dict[str(id)]
                id_seg_mask = coors_to_mask(coors, image_shape)
                split_mask = (consensus_fail * id_seg_mask)
                segmentation[split_mask] = num_group + id
        """
    return segmentation


segs_path_test = '/home/xiaotx/2017EXBB/inf_whole/part5/'

merge_dict_save_path_test = segs_path_test
merge_dict_path_test = segs_path_test + 'merge_dict.pkl'

segs_code_dict = convert_coor_all(segs_path_test)
merge_dict = merge(segs_path_test, segs_code_dict, 0.20)
print(len(merge_dict))


pickle_obj(merge_dict, 'merge_dict', merge_dict_save_path_test)
merge_dict_test = load_obj(merge_dict_path_test)


merge_group_dict_test = merge_segs(merge_dict_test)

image_shape = (160, 5000, 1887)

segmentation = segs_reconstructor(segs_path_test, merge_group_dict_test, image_shape, cons_thr=1)
RGB_img = segs_to_RGB(segmentation)

save_path = merge_dict_save_path_test + 'test_part5.tif'
skimage.io.imsave(save_path, RGB_img.astype('uint8'))


