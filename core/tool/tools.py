"""Helper functions."""
import numpy as np
from pathlib import Path
import natsort
import skimage
import pickle
import h5py
import os
from typing import Dict, Tuple, List
import ast


def sort_files(dir_path: str) -> List:
    """Sorts and returns a list of files from a directory path"""
    entries = Path(dir_path)
    files = []
    for entry in entries.iterdir():
        files.append(entry.name)
    sorted_files = natsort.natsorted(files, reverse=False)
    return sorted_files


def pickle_obj(obj: Dict, name: str, path: str) -> None:
    """Save dict as a pickle object"""
    file_path = os.path.join(path, name, '.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(file_path: str) -> Dict:
    """Load the pickle object from file_path."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_raw_image(path: str, group: str = '/image') -> Tuple:
    """Load the raw image volume."""
    file_extension = os.path.splitext(path)[1]
    if file_extension == '.h5':
        with h5py.File(path, 'r') as f:
            images = (f[group][()].astype(np.float32) - 128) / 33
    else:
        images = ((skimage.io.imread(path)).astype(np.float32) - 128) / 33
    return images


def resume_dict_load(dict_path: str, name: str, resume_obj: int) -> Dict:
    """Load dict to resume processing."""
    dict_path = os.path.join(dict_path, name, '.pkl')
    if os.path.exists(dict_path):
        resume = load_obj(dict_path)
    else:
        resume = {'resume_seed': resume_obj}
    resume_obj = resume['resume_seed']
    print(f'Resume {resume_obj}')
    return resume


def resume_re_segd_count_mask(file_path: str, shape: Tuple) -> Tuple:
    """Load mask file if it exists otherwise create a new one."""
    file_path = os.path.join(file_path, 're_seged_count_mask.tif')
    if os.path.exists(file_path):
        re_seged_count_mask = skimage.io.imread(file_path)
    else:
        re_seged_count_mask = np.zeros(shape, dtype=np.uint8)
    return re_seged_count_mask


def load_seeds_from_file(seeds_file_path: str, manual_seed: bool) -> Dict:
    """
    Load seeds either from a .h5 (auto generated seeds)
    or from a .txt (manually specified seeds) file.
    """
    seeds = {}
    if os.path.exists(seeds_file_path):
        if not manual_seed:
            with h5py.File(seeds_file_path, 'r') as f:
                seeds = f['seeds'][()]
                seeds = list(seeds)
                seeds = {i + 1: coord for i, coord in enumerate(seeds)}
        else:
            # Expects a .txt file of seed coordinates in a Dict format
            # Each line as {id: [z, y, x]}
            with open(seeds_file_path, 'r') as f:
                contents = f.read()
                seeds = ast.literal_eval(contents)
    else:
        print(f'{seeds_file_path} - seeds file does not exist!')

    return seeds


def z_resize_no_inter(data,factor):
    """TODO: Add function description and type hints."""
    resized = skimage.transform.resize(data,
                                       (data.shape[0] * factor, data.shape[1],
                                        data.shape[2]),
                                       mode='edge',
                                       anti_aliasing=False,
                                       anti_aliasing_sigma=None,
                                       preserve_range=True,
                                       order=0)
    return resized
