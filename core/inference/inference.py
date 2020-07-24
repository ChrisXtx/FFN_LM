"""Tracks inference progress and manages inference related policies."""
from scipy.special import expit
from scipy.special import logit
import torch
import os
import numpy as np
import h5py
import skimage.feature
import weakref
from collections import deque
from torch.autograd import Variable
import skimage.io
import threading
from typing import Tuple, List

MAX_SELF_CONSISTENT_ITERS = 32


def quantize_probability(probability_map: np.ndarray) -> np.ndarray:
    """Quantizes a probability map into a byte array."""
    quantized = np.digitize(probability_map, np.linspace(0.0, 1.0, 255))

    # Digitize never uses the 0-th bucket.
    quantized[np.isnan(probability_map)] = 0
    return quantized.astype(np.uint8)


def get_scored_move_offsets(deltas: np.ndarray, probability_map: np.ndarray,
                            flex_faces: int, threshold: float = 0.8) \
                            -> [float, Tuple]:
    """Looks for potential moves for a FFN.
    The possible moves are determined by extracting probability map values
    corresponding to cuboid faces at +/- deltas, and considering the highest
    probability value for every face.
    Args:
      deltas: (z,y,x) tuple of base move offsets for the 3 axes
      probability_map: current probability map as a (z,y,x) numpy array
      threshold: minimum score required at the new FoV center for a move to be
          considered valid
      flex_faces : multipliers of the faces
    Yields:
      tuples of:
        score (probability at the new FoV center),
        position offset tuple (z,y,x) relative to center of prob_map
      The order of the returned tuples is arbitrary and should not be depended
      upon. In particular, the tuples are not necessarily sorted by score.
    """
    flex_deltas = [deltas]
    if flex_faces == 0:
        pass
    else:
        for flex in range(1, flex_faces):
            # TODO: define movable faces
            if (deltas[0] % flex * 2) != 0:
                deltas_down = (deltas + (deltas[0] % flex * 2)) / (flex*2)
            else:
                deltas_down = deltas / (flex*2)
            deltas_half_up = deltas + deltas_down
            deltas_up = deltas * 2 * flex
            deltas_down = [int(deltas_down[0]), int(deltas_down[1]),
                           int(deltas_down[2])]
            deltas_up = [int(deltas_up[0]), int(deltas_up[1]),
                         int(deltas_up[2])]
            deltas_half_up = [int(deltas_half_up[0]), int(deltas_half_up[1]),
                              int(deltas_half_up[2])]
            flex_deltas.append(deltas_up)
            flex_deltas.append(deltas_down)
            flex_deltas.append(deltas_half_up)

    for deltas in flex_deltas:
        center = np.array(probability_map.shape) // 2
        assert center.size == 3
        # Selects a working sub_volume no more than +/- delta away
        # from the current center point.
        sub_volume = [slice(c - dx, c + dx + 1) for c, dx
                      in zip(center, deltas)]

        done = set()
        for axis, axis_delta in enumerate(deltas):
            if axis_delta == 0:
                continue
            for axis_offset in (-axis_delta, axis_delta):
                # Move exactly by the delta along the current axis,
                # then select the face of the sub_volume orthogonal to the
                # current axis.
                selected_face = sub_volume[:]
                selected_face[axis] = axis_offset + center[axis]
                face_probability = probability_map[tuple(selected_face)]
                face_probability_shape = face_probability.shape

                # Find voxel with maximum activation.
                face_pos = np.unravel_index(face_probability.argmax(),
                                            face_probability_shape)
                score = face_probability[face_pos]

                # Only move if activation crosses threshold.
                if score < threshold:
                    continue

                # Convert within-face position to be relative
                # vs the center of the face.
                relative_pos = [face_pos[0] - face_probability_shape[0] // 2,
                                face_pos[1] - face_probability_shape[1] // 2]
                relative_pos.insert(axis, axis_offset)

                ret = (score, tuple(relative_pos))

                if ret not in done:
                    done.add(ret)
                    yield ret


class BaseMovementPolicy(object):
    """Base class for movement policy queues.
    The principal usage is to initialize once with the policy's parameters and
    set up a queue for candidate positions. From this queue candidates can be
    iteratively consumed and the scores should be updated in the FFN
    segmentation loop.
    """

    def __init__(self, canvas: 'Canvas', scored_coords: deque,
                 deltas: Tuple):
        """Initializes the policy.
        Args:
          canvas: Canvas object for FFN inference
          scored_coords: mutable container of tuples (score, zyx coord)
          deltas: step sizes as (z,y,x)
        """
        # TODO: Remove circular reference between Canvas and seed policies.
        self.canvas = weakref.proxy(canvas)
        self.scored_coords = scored_coords
        self.deltas = np.array(deltas)

    def __len__(self):
        return len(self.scored_coords)

    def __iter__(self):
        return self

    def next(self):
        """TODO: Description."""
        raise StopIteration()

    def append(self, item: Tuple):
        """TODO:  Description."""
        self.scored_coords.append(item)

    def update(self, probability_map: np.ndarray, position: Tuple):
        """Updates the state after an FFN inference call.
        Args:
          probability_map: object probability map returned
                           by the FFN (in logit space)
          position: position of the center of the FoV where inference was
                    performed (z, y, x).
        """
        raise NotImplementedError()

    def get_state(self):
        """Returns the state of this policy as a pickable Python object."""
        raise NotImplementedError()

    def restore_state(self, state):
        """TODO: Description."""
        raise NotImplementedError()

    def reset_state(self, start_pos: Tuple):
        """Resets the policy.
        Args:
          start_pos: starting position of the current object as z, y, x
        """
        raise NotImplementedError()


class FaceMaxMovementPolicy(BaseMovementPolicy):
    """Selects candidates from maxima on prediction cuboid faces."""

    def __init__(self, canvas: 'Canvas', deltas: Tuple = (4, 8, 8),
                 score_threshold: float = 0.8):
        self.done_rounded_coords = set()
        self.score_threshold = score_threshold
        self._start_position = None
        super().__init__(canvas, deque([]), deltas)

    def reset_state(self, start_position: Tuple):
        """TODO: Description."""
        self.scored_coords = deque([])
        self.done_rounded_coords = set()
        self._start_position = start_position

    def get_state(self):
        """TODO: Description and type hints."""
        return [(self.scored_coords, self.done_rounded_coords)]

    def restore_state(self, state):
        """TODO: Description and type hints."""
        self.scored_coords, self.done_rounded_coords = state[0]

    def __next__(self) -> Tuple:
        """Pops positions from queue until a valid one is found
        and returns it."""
        while self.scored_coords:
            _, coord = self.scored_coords.popleft()
            coord = tuple(coord)
            if self.quantize_position(coord) in self.done_rounded_coords:
                continue
            if self.canvas.is_valid_pos(coord):
                break
        else:  # Else goes with while, not with if!
            # FIXME: misaligned else
            raise StopIteration()

        return tuple(coord)

    def next(self):
        """TODO: Description."""
        return self.__next__()

    def quantize_position(self, position: Tuple):
        """Quantizes the positions symmetrically to a grid
        downsampled by deltas."""
        # Compute offset relative to the origin of the current segment and
        # shift by half delta size. This ensures that all directions are
        # treated approximately symmetrically -- i.e. the origin point lies
        # in the middle of a cell of the quantized lattice, as opposed to a
        # corner of that cell.
        rel_pos = (np.array(position) - self._start_position)
        coord = (rel_pos + self.deltas//2) // np.maximum(self.deltas, 1)
        return tuple(coord)

    def update(self, probability_map: np.ndarray, position: Tuple, flex: int):
        """Adds movements to queue for the cuboid face maxima
        of ``probability_map``."""
        # TODO: Fix override 'flex' warning
        qpos = self.quantize_position(position)
        self.done_rounded_coords.add(qpos)

        scored_coords = get_scored_move_offsets(self.deltas, probability_map,
                                                flex,
                                                threshold=self.score_threshold)
        scored_coords = sorted(scored_coords, reverse=True)
        for score, rel_coord in scored_coords:
            # Convert to whole cube coordinates
            coord = [rel_coord[i] + position[i] for i in range(3)]
            self.scored_coords.append((score, coord))


class Canvas(object):
    """Runs inference within a subvolume and writes out segmentations."""

    def __init__(self, model: 'FFN', images: np.ndarray, size: Tuple,
                 delta: Tuple, seg_thr: float, mov_thr: float,
                 act_thr: float, flex_faces: int, re_seg_thr: int,
                 vox_thr: int, data_save_path: str, re_seg_mask: np.ndarray,
                 save_chunk: int, resume_seed: int, manual_seed: bool,
                 process_id: int):

        self.process_id = process_id
        self.manual_seed = manual_seed
        self.model = model
        self.images = images

        self.shape = images.shape[:-1]
        self.input_size = np.array(size)
        self.margin = np.array(size) // 2
        self.seg_thr = logit(seg_thr)
        self.mov_thr = logit(mov_thr)
        self.act_thr = logit(act_thr)
        self.flex_faces = flex_faces
        self.re_seg_thr = re_seg_thr

        self.data_save_path = data_save_path
        self.save_count = 1
        self.re_seg_mask = re_seg_mask
        self.save_chunk = save_chunk
        self.resume_seed = resume_seed
        self.save_part = 1 + int(self.resume_seed / self.save_chunk)

        self.seed = np.zeros(self.shape, dtype=np.float32)
        self.seg_prob_i = np.zeros(self.shape, dtype=np.uint8)

        self.vox_thr = vox_thr
        self.target_dic = {}

        self.seed_policy = None
        self.max_id = 0
        # Maps of segment id -> ..
        self.origins = {}  # seed location
        self.overlaps = {}  # (ids, number overlapping voxels)

        self.movement_policy = FaceMaxMovementPolicy(self, deltas=delta,
                                                     score_threshold=mov_thr)
        self.reset_state((0, 0, 0))
        self.history = None
        self.history_deleted = None
        self._min_pos = None
        self._max_pos = None

    def init_seed(self, position: Tuple):
        """Reinitiailizes the object mask with a seed.
        Args:
          position: position at which to place the seed (z, y, x)
        """
        self.seed[...] = np.nan
        self.seed[position] = self.act_thr

    def reset_state(self, start_position: Tuple):
        """Resetting the movement_policy is currently necessary to update the
        policy's bitmask for whether a position is already segmented (the
        canvas updates the segmented mask only between calls to segment_at
        and therefore the policy does not update this mask for every call.).
        """
        self.movement_policy.reset_state(start_position)
        self.history = []
        self.history_deleted = []

        self._min_pos = np.array(start_position)
        self._max_pos = np.array(start_position)

    def is_valid_pos(self, position: Tuple,
                     ignore_move_threshold: bool = False) -> bool:
        """Returns True if segmentation should be attempted at the
        given position.
        Args
          position: position to check as (z, y, x)
          ignore_move_threshold: (boolean) when starting a new segment at
                position the move threshold can and must be ignored.
        Returns:
          Boolean indicating whether to run FFN inference at the given
          position.
        """

        if not ignore_move_threshold:
            if self.seed[position] < self.mov_thr:
                return False

        # Not enough image context?
        np_pos = np.array(position)
        low = np_pos - self.margin
        high = np_pos + self.margin

        if np.any(low < 0) or np.any(high >= self.shape):
            return False

        # Location already segmented?
        # if self.segmentation[pos] > 0:
        # return False

        return True

    def predict(self, position: Tuple) -> [np.ndarray, np.ndarray]:
        """Runs a single step of FFN prediction."""
        # Top-left corner of the FoV.
        start = np.array(position) - self.margin
        end = start + self.input_size
        assert np.all(start >= 0)

        # Crop the raw image as input.
        images_pred = self.images[start[0]:end[0],
                                  start[1]:end[1],
                                  start[2]:end[2], :]
        images_pred = images_pred.transpose(3, 0, 1, 2)
        seeds = self.seed[start[0]:end[0],
                          start[1]:end[1],
                          start[2]:end[2]].copy()

        init_prediction = np.isnan(seeds)
        seeds[init_prediction] = np.float32(logit(0.05))
        images_pred = torch.from_numpy(images_pred).float().unsqueeze(0)
        seeds = torch.from_numpy(seeds).float().unsqueeze(0).unsqueeze(0)

        input_data = torch.cat([images_pred, seeds], dim=1)
        input_data = Variable(input_data.cuda())

        # Model inference.
        logits = self.model(input_data)
        updated = (seeds.cuda() + logits).detach().cpu().numpy()
        prob = expit(updated)

        return np.squeeze(prob), np.squeeze(updated)

    def update_at(self, position: Tuple) -> [np.ndarray, List]:
        """Updates object mask prediction at a specific position."""
        # FIXME: where is global old_err defined - possibly remove
        global old_err
        off = self.input_size // 2  # zyx
        start = np.array(position) - off
        end = start + self.input_size

        sel = [slice(s, e) for s, e in zip(start, end)]

        logit_seed = np.array(self.seed[tuple(sel)])
        init_prediction = np.isnan(logit_seed)
        logit_seed[init_prediction] = np.float32(logit(0.05))

        for _ in range(MAX_SELF_CONSISTENT_ITERS):
            # Model inference
            probabilities, logits = self.predict(position)
            break

        # Update seed.
        sel = [slice(s, e) for s, e in zip(start, end)]

        # Bias towards over-segmentation by making it impossible to reverse
        # disconnectedness predictions in the course of inference.
        th_max = logit(0.5)
        old_seed = self.seed[tuple(sel)]

        if np.mean(logits >= self.mov_thr) > 0:
            # Because (x > NaN) is always False, this mask excludes positions
            # that were previously uninitialized (i.e. set to NaN in old_seed).
            try:
                old_err = np.seterr(invalid='ignore')
                mask = ((old_seed < th_max) & (logits > old_seed))
            finally:
                np.seterr(**old_err)
            logits[mask] = old_seed[mask]

        # Update working space.
        self.seed[tuple(sel)] = logits

        return logits, sel

    def segment_at(self, start_pos: Tuple, process_id: int, tag: str) \
            -> bool:
        """TODO: Description."""
        t_lock = threading.Lock()

        try:
            if not self.is_valid_pos(start_pos, ignore_move_threshold=True):
                return False

            # Check if the seed location have been segmented many times
            if self.re_seg_mask[start_pos] >= self.re_seg_thr:
                print(f'Skip {process_id}')
                return False

            self.seg_prob_i = np.zeros(self.shape, dtype=np.uint8)
            self.seed = np.zeros(self.shape, dtype=np.float32)
            self.init_seed(start_pos)
            self.reset_state(start_pos)

            if not self.movement_policy:
                # Add first element with arbitrary priority 1 (it will be
                # consumed right away anyway).
                item = (self.movement_policy.score_threshold * 2, start_pos)
                self.movement_policy.append(item)

            step_iter = 0

            for pos in self.movement_policy:
                # Terminate early if the seed got too weak.
                if self.seed[start_pos] < self.mov_thr:
                    break

                # Stepping
                pred, sel_i = self.update_at(pos)
                step_iter += 1
                print(f'Process ID {process_id}, Step {step_iter}')
                # Updated cube fov size
                sel_i_s = sel_i

                # Manual seed: save image out each step
                if self.manual_seed:
                    if not os.path.exists('./manual_seed_data/'):
                        os.makedirs('./manual_seed_data/')
                    try:
                        mask = self.seed[tuple(sel_i_s)] >= self.seg_thr
                        self.seg_prob_i[tuple(sel_i_s)][mask] = \
                            quantize_probability(
                                expit(self.seed[tuple(sel_i_s)][mask]))
                    except RuntimeError:
                        return False

                    # Save the predicted mask out for each step
                    # TODO: Save out to user input location
                    skimage.io.imsave(
                        './data/FFN_object_inf_{}_step{}.tif'.format(
                            process_id, step_iter), self.seg_prob_i)

                # Update valid locations for movement
                self.movement_policy.update(pred, pos, flex=self.flex_faces)
                assert np.all(pred.shape == self.input_size)

            try:
                mask = (self.seed >= self.seg_thr)
                self.seg_prob_i[mask] = quantize_probability(
                                        expit(self.seed[mask]))
            except RuntimeError:
                return False

            mask_seg_prob_i = (self.seg_prob_i >= 128)
            if np.sum(mask_seg_prob_i) >= self.vox_thr:
                self.re_seg_mask[mask_seg_prob_i] += 1

                # Save segmentation from each seed
                seg_prob_coords = np.argwhere(mask_seg_prob_i).astype('int16')
                t_lock.acquire()
                self.save_count += 1
                id_save = '{}'.format(process_id)
                print(f'Save count: {self.save_count}, '
                      f'Process ID: {self.process_id}')

                # Save the segmentation by part
                if self.save_count % self.save_chunk == 0:
                    self.save_part += 1

                # Update re_seg_mask
                if self.save_count % 2000 == 0:
                    skimage.io.imsave(self.data_save_path + 're_seg_mask.tif',
                                      self.re_seg_mask.astype('uint8'))

                print(f'Segmentation saved! '
                      f'seed:{process_id}, '
                      f'coord:{start_pos}, '
                      f'completed num: {self.save_count}')
                try:
                    name = f"{tag}seg_of_seeds_test_part{self.save_part}.h5"
                    file_path = os.path.join(self.data_save_path, name)
                    print(f'SavePath {file_path}')
                    with h5py.File(file_path, 'a') as f:
                        f.create_dataset(id_save, data=seg_prob_coords,
                                         compression='gzip')
                except OSError:
                    return True
                t_lock.release()

                return True
            else:
                print(f'Too small {process_id}')

        except RuntimeError:
            return False
