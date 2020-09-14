"""Find pairs of matching masks from two folders of segmentation masks (
ground truth vs automated) and calculate precision, recall, fscore."""
import argparse
import os
from scipy.spatial import cKDTree
from typing import Dict
from skimage import io
import numpy as np


def parse():
    """Argument Parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth_path', type=str, required=True)
    parser.add_argument('--prediction_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.15)
    args = parser.parse_args()
    return args


def load_mask_files(mask_files_path: str) -> Dict:
    """"Load mask files and read coordinates from each."""
    coords = {}
    files = os.listdir(mask_files_path)

    for file in files:
        file_path = os.path.join(mask_files_path, file)
        if file != ".DS_Store":
            image = io.imread(file_path)
            head, tail = os.path.split(file)
            coords[tail] = (np.argwhere(image)).astype('uint32')
    return coords


def create_kdtree(coords: Dict) -> Dict:
    """From a Dict of coordinates create ckdtrees."""
    trees = {}
    for file, coord in coords.items():
        trees[file] = cKDTree(coord)
    return trees


def find_matching_mask_pairs(ground_truth_coord_files: Dict,
                             pred_coord_files: Dict, threshold: float) -> Dict:
    """"From two files->coords dictionaries, find matching ones"""
    # Create cKDtrees
    print("Creating ckdtrees...")
    ground_truth_kdtrees = create_kdtree(ground_truth_coord_files)
    pred_kdtrees = create_kdtree(pred_coord_files)

    matches = {}
    # Find the best match for each tree by finding the max overlap with ground
    # truth segmentation.
    print("Finding matches and calculating metrics...")
    for gt_file, gt_tree in ground_truth_kdtrees.items():
        best_match = ()
        min_overlap = 0
        for pred_file, pred_tree in pred_kdtrees.items():
            overlap = gt_tree.query_ball_tree(pred_tree, r=threshold)
            non_empty_neighbors = [neighbor[0] for neighbor in overlap if
                                   neighbor]
            neighbor_count = len(non_empty_neighbors)

            if neighbor_count > min_overlap:
                min_overlap = neighbor_count
                precision = neighbor_count / len(pred_coord_files[pred_file])
                recall = neighbor_count / len(ground_truth_coord_files[
                                                  gt_file])
                fscore = 0
                if recall + precision > 0:
                    fscore = 2 * recall * precision / (recall + precision)
                best_match = (pred_file, neighbor_count, precision, recall,
                              fscore)

        matches[gt_file] = best_match

    return matches


def run() -> None:
    """Steps for finding matches and calculating precison, recall & fscore
    between two segmentations for which individual process masks have
    already been created."""
    args = parse()

    # Load files from two folders and get the coordinates as Dict for each
    print("Loading files and extracting coordinates...")
    ground_truth_coord_files = load_mask_files(args.ground_truth_path)
    prediction_coord_files = load_mask_files(args.prediction_path)
    threshold = args.threshold

    # Get mask matches as a Dict
    matches = find_matching_mask_pairs(ground_truth_coord_files,
                                       prediction_coord_files, threshold)
    print(matches)

    # Write matches to file
    print("Writing matches and metrics file...")
    with open(args.save_path, "w") as f:
        f.write(str(matches))

    print("Done.")


if __name__ == "__main__":
    run()
