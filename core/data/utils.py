

import itertools
import sys
from scipy.special import expit
from scipy.special import logit
import torch
import six
import numpy as np
from scipy import ndimage
import random
import skimage.feature
import logging
import weakref
from collections import namedtuple
from collections import deque
import time
from torch.autograd import Variable
import cv2
import skimage
from scipy.stats import stats
import os
import h5py
import threading
import inference_run

MAX_SELF_CONSISTENT_ITERS = 32
HALT_SILENT = 0
PRINT_HALTS = 1
HALT_VERBOSE = 2

OriginInfo = namedtuple('OriginInfo', ['start_zyx', 'iters', 'walltime_sec'])
HaltInfo = namedtuple('HaltInfo', ['is_halt', 'extra_fetches'])

# multithreads communication

save_count = 1
re_seged_count_mask = np.zeros(inference_run.images.shape[:-1], dtype = np.uint8)
save_chunk = inference_run.args.save_chunk
resume_seed = inference_run.args.resume_seed
save_part = 1 + int(resume_seed/save_chunk)
images = inference_run.images


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_seed(shape, pad=0.05, seed=0.95):
    """seed——init"""
    seed_array = np.full(list(shape), pad, dtype=np.float32)
    idx = tuple([slice(None)] + list(np.array(shape) // 2))
    seed_array[idx] = seed
    return seed_array


def fixed_offsets(seed, fov_moves, threshold=0.8):
    """offsets to deltas."""
    for off in itertools.chain([(0, 0, 0)], fov_moves):
        is_valid_move = seed[0,
                             seed.shape[1] // 2 + off[2],
                             seed.shape[2] // 2 + off[1],
                             seed.shape[3] // 2 + off[0]
                        ] >= logit(np.array(threshold))

        if not is_valid_move:
            continue

        yield off


def center_crop_and_pad(data, coor, target_shape):
    """crop patch based on center coordinate"""
    target_shape = np.array(target_shape)

    start = coor - target_shape // 2
    end = start + target_shape

    # assert np.all(start >= 0)

    selector = [slice(s, e) for s, e in zip(start, end)]
    cropped = data[tuple(selector)]

    if target_shape is not None:

        if len(cropped.shape) > 3:
            target_shape = np.array(target_shape)
            delta = target_shape - cropped.shape[:-1]
            pre = delta // 2
            post = delta - delta // 2

            paddings = []  # no padding for batch
            paddings.extend(zip(pre, post))
            paddings.append((0, 0))
            cropped = np.pad(cropped, paddings, mode='constant')
        else:
            target_shape = np.array(target_shape)
            delta = target_shape - cropped.shape
            pre = delta // 2
            post = delta - delta // 2

            paddings = []  # no padding for batch
            paddings.extend(zip(pre, post))
            cropped = np.pad(cropped, paddings, mode='constant')

    return cropped


def crop_and_pad(data, offset, crop_shape, target_shape=None):
    """crop patch based on offset"""
    # Spatial dimensions only. All vars in zyx.
    shape = np.array(data.shape[1:])
    crop_shape = np.array(crop_shape)
    offset = np.array(offset[::-1])

    start = shape // 2 - crop_shape // 2 + offset
    end = start + crop_shape

    assert np.all(start >= 0)

    selector = [slice(s, e) for s, e in zip(start, end)]
    selector = tuple([slice(None)] + selector)
    cropped = data[selector]

    if target_shape is not None:
        target_shape = np.array(target_shape)
        delta = target_shape - crop_shape
        pre = delta // 2
        post = delta - delta // 2

        paddings = [(0, 0)]  # no padding for batch
        paddings.extend(zip(pre, post))
        paddings.append((0, 0))  # no padding for channels

        cropped = np.pad(cropped, paddings, mode='constant')

    return cropped


def get_example(loader, shape, get_offsets):
    while True:
        iteration, (image, targets, seed, coor) = next(enumerate(loader))
        seed = seed.numpy().copy()
        for off in get_offsets(seed):
            predicted = crop_and_pad(seed, off, shape)[np.newaxis, ...]
            patches = crop_and_pad(image.squeeze(), off, shape).unsqueeze(0)
            labels = crop_and_pad(targets, off, shape).unsqueeze(0)
            offset = off
            assert predicted.base is seed
            yield predicted, patches, labels, offset


def get_batch(loader, batch_size, shape, get_offsets):
    def _batch(iterable):
        for batch_vals in iterable:
            yield zip(*batch_vals)

    for seeds, patches, labels, offsets in _batch(six.moves.zip(
            *[get_example(loader, shape, get_offsets) for _
              in range(batch_size)])):

        batched_seeds = np.concatenate(seeds)

        yield (batched_seeds, torch.cat(patches, dim=0).float(), \
               torch.cat(labels, dim=0).float(), offsets)

        for i in range(batch_size):
            seeds[i][:] = batched_seeds[i, ...]


def update_seed(updated, seed, model, pos):
    start = pos - model.input_size // 2
    end = start + model.input_size
    assert np.all(start >= 0)

    selector = [slice(s, e) for s, e in zip(start, end)]
    seed[selector] = np.squeeze(updated)


def no_halt(verbosity=HALT_SILENT, log_function=logging.info):
    """Dummy HaltInfo."""

    def _halt_signaler(*unused_args, **unused_kwargs):
        return False

    def _halt_signaler_verbose(fetches, pos, **unused_kwargs):
        log_function('%s, %s' % (pos, fetches))
        return False

    if verbosity == HALT_VERBOSE:
        return HaltInfo(_halt_signaler_verbose, [])
    else:
        return HaltInfo(_halt_signaler, [])


def self_prediction_halt(
        threshold, orig_threshold=None, verbosity=HALT_SILENT,
        log_function=logging.info):
    """HaltInfo based on FFN self-predictions."""

    def _halt_signaler(fetches, pos, orig_pos, counters, **unused_kwargs):
        """Returns true if FFN prediction should be halted."""
        if pos == orig_pos and orig_threshold is not None:
            t = orig_threshold
        else:
            t = threshold

        # [0] is by convention the total incorrect proportion prediction.
        halt = fetches['self_prediction'][0] > t

        if halt:
            counters['halts'].Increment()

        if verbosity == HALT_VERBOSE or (
                halt and verbosity == PRINT_HALTS):
            log_function('%s, %s' % (pos, fetches))

        return halt

    # Add self_prediction to the extra_fetches.
    return HaltInfo(_halt_signaler, ['self_prediction'])


def quantize_probability(prob):
    """Quantizes a probability map into a byte array."""
    ret = np.digitize(prob, np.linspace(0.0, 1.0, 255))

    # Digitize never uses the 0-th bucket.
    ret[np.isnan(prob)] = 0
    return ret.astype(np.uint8)


def get_scored_move_offsets(deltas, prob_map, threshold=0.8, flex_faces=1):
    """Looks for potential moves for a FFN.

    The possible moves are determined by extracting probability map values
    corresponding to cuboid faces at +/- deltas, and considering the highest
    probability value for every face.

    Args:
      deltas: (z,y,x) tuple of base move offsets for the 3 axes
      prob_map: current probability map as a (z,y,x) numpy array
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
        for flex in range(flex_faces):
            if flex == 0:
                continue
            if (deltas[0] % flex * 2) != 0:
                deltas_down = (deltas + (deltas[0] % flex * 2)) / flex * 2
            else:
                deltas_down = deltas / (flex * 2)
            deltas_half_up = deltas + deltas_down
            deltas_up = deltas * 2 * flex
            deltas_down = [int(deltas_down[0]), int(deltas_down[1]), int(deltas_down[2])]
            deltas_up = [int(deltas_up[0]), int(deltas_up[1]), int(deltas_up[2])]
            deltas_half_up = [int(deltas_half_up[0]), int(deltas_half_up[1]), int(deltas_half_up[2])]
            flex_deltas.append(deltas_up)
            flex_deltas.append(deltas_down)
            flex_deltas.append(deltas_half_up)

    for deltas in flex_deltas:
        center = np.array(prob_map.shape) // 2
        assert center.size == 3
        # Selects a working subvolume no more than +/- delta away from the current
        # center point.
        subvol_sel = [slice(c - dx, c + dx + 1) for c, dx
                      in zip(center, deltas)]

        done = set()
        for axis, axis_delta in enumerate(deltas):
            if axis_delta == 0:
                continue
            for axis_offset in (-axis_delta, axis_delta):
                # Move exactly by the delta along the current axis, and select the face
                # of the subvolume orthogonal to the current axis.
                face_sel = subvol_sel[:]
                face_sel[axis] = axis_offset + center[axis]
                face_prob = prob_map[tuple(face_sel)]
                shape = face_prob.shape

                # Find voxel with maximum activation.
                face_pos = np.unravel_index(face_prob.argmax(), shape)
                score = face_prob[face_pos]

                # Only move if activation crosses threshold.
                if score < threshold:
                    continue

                # Convert within-face position to be relative vs the center of the face.
                relative_pos = [face_pos[0] - shape[0] // 2, face_pos[1] - shape[1] // 2]
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

    def __init__(self, canvas, scored_coords, deltas):
        """Initializes the policy.

        Args:
          canvas: Canvas object for FFN inference
          scored_coords: mutable container of tuples (score, zyx coord)
          deltas: step sizes as (z,y,x)
        """
        # TODO(mjanusz): Remove circular reference between Canvas and seed policies.
        self.canvas = weakref.proxy(canvas)
        self.scored_coords = scored_coords
        self.deltas = np.array(deltas)

    def __len__(self):
        return len(self.scored_coords)

    def __iter__(self):
        return self

    def next(self):
        raise StopIteration()

    def append(self, item):
        self.scored_coords.append(item)

    def update(self, prob_map, position):
        """Updates the state after an FFN inference call.

        Args:
          prob_map: object probability map returned by the FFN (in logit space)
          position: postiion of the center of the FoV where inference was performed
              (z, y, x)
        """
        raise NotImplementedError()

    def get_state(self):
        """Returns the state of this policy as a pickable Python object."""
        raise NotImplementedError()

    def restore_state(self, state):
        raise NotImplementedError()

    def reset_state(self, start_pos):
        """Resets the policy.

        Args:
          start_pos: starting position of the current object as z, y, x
        """
        raise NotImplementedError()


class FaceMaxMovementPolicy(BaseMovementPolicy):
    """Selects candidates from maxima on prediction cuboid faces."""

    def __init__(self, canvas, deltas=(4, 8, 8), score_threshold=0.8):
        self.done_rounded_coords = set()
        self.score_threshold = score_threshold
        self._start_pos = None
        super(FaceMaxMovementPolicy, self).__init__(canvas, deque([]), deltas)

    def reset_state(self, start_pos):
        self.scored_coords = deque([])
        self.done_rounded_coords = set()
        self._start_pos = start_pos

    def get_state(self):
        return [(self.scored_coords, self.done_rounded_coords)]

    def restore_state(self, state):
        self.scored_coords, self.done_rounded_coords = state[0]

    def __next__(self):
        """Pops positions from queue until a valid one is found and returns it."""
        while self.scored_coords:
            _, coord = self.scored_coords.popleft()
            coord = tuple(coord)
            if self.quantize_pos(coord) in self.done_rounded_coords:
                continue
            if self.canvas.is_valid_pos(coord):
                break
        else:  # Else goes with while, not with if!
            raise StopIteration()

        return tuple(coord)

    def next(self):
        return self.__next__()

    def quantize_pos(self, pos):
        """Quantizes the positions symmetrically to a grid downsampled by deltas."""
        # Compute offset relative to the origin of the current segment and
        # shift by half delta size. This ensures that all directions are treated
        # approximately symmetrically -- i.e. the origin point lies in the middle of
        # a cell of the quantized lattice, as opposed to a corner of that cell.
        rel_pos = (np.array(pos) - self._start_pos)
        coord = (rel_pos + self.deltas // 2) // np.maximum(self.deltas, 1)
        return tuple(coord)

    def update(self, prob_map, position):
        """Adds movements to queue for the cuboid face maxima of ``prob_map``."""
        qpos = self.quantize_pos(position)
        self.done_rounded_coords.add(qpos)

        scored_coords = get_scored_move_offsets(self.deltas, prob_map,
                                                threshold=self.score_threshold, flex_faces=1)
        scored_coords = sorted(scored_coords, reverse=True)
        for score, rel_coord in scored_coords:
            # convert to whole cube coordinates
            coord = [rel_coord[i] + position[i] for i in range(3)]
            self.scored_coords.append((score, coord))


class Canvas(object):



    def __init__(self, model, size, delta, seg_thr, mov_thr, act_thr, re_seg_thr, vox_thr, data_save_path,
                 process_id):


        self.process_id = process_id
        self.model = model

        self.shape = images.shape[:-1]
        self.input_size = np.array(size)
        self.margin = np.array(size) // 2
        self.seg_thr = logit(seg_thr)
        self.mov_thr = logit(mov_thr)
        self.act_thr = logit(act_thr)
        self.re_seg_thr = re_seg_thr

        self.data_save_path = data_save_path

        self.seed = np.zeros(self.shape, dtype=np.float32)
        self.seg_prob_i = np.zeros(self.shape, dtype=np.uint8)  # temp save out (pred mask) for each step


        self.vox_thr = vox_thr
        self.target_dic = {}

        self.seed_policy = None
        self.max_id = 0
        # Maps of segment id -> ..
        self.origins = {}  # seed location
        self.overlaps = {}  # (ids, number overlapping voxels)

        self.movement_policy = FaceMaxMovementPolicy(self, deltas=delta, score_threshold=self.mov_thr)
        self.reset_state((0, 0, 0))

    def init_seed(self, pos):
        """Reinitiailizes the object mask with a seed.

        Args:
          pos: position at which to place the seed (z, y, x)
        """
        self.seed[...] = np.nan
        self.seed[pos] = self.act_thr

    def reset_state(self, start_pos):
        # Resetting the movement_policy is currently necessary to update the
        # policy's bitmask for whether a position is already segmented (the
        # canvas updates the segmented mask only between calls to segment_at
        # and therefore the policy does not update this mask for every call.).
        self.movement_policy.reset_state(start_pos)
        self.history = []
        self.history_deleted = []

        self._min_pos = np.array(start_pos)
        self._max_pos = np.array(start_pos)

    def is_valid_pos(self, pos, ignore_move_threshold=False):
        """Returns True if segmentation should be attempted at the given position.

        Args:
          pos: position to check as (z, y, x)
          ignore_move_threshold: (boolean) when starting a new segment at pos the
              move threshold can and must be ignored.

        Returns:
          Boolean indicating whether to run FFN inference at the given position.
        """

        if not ignore_move_threshold:
            if self.seed[pos] < self.mov_thr:
                return False

        # Not enough image context?
        np_pos = np.array(pos)
        low = np_pos - self.margin
        high = np_pos + self.margin

        if np.any(low < 0) or np.any(high >= self.shape):
            return False

        # Location already segmented?
        # if self.segmentation[pos] > 0:
        # return False

        return True

    def predict(self, pos):
        """Runs a single step of FFN prediction.
        """
        global images
        # Top-left corner of the FoV.
        start = np.array(pos) - self.margin
        end = start + self.input_size

        assert np.all(start >= 0)

        # selector = [slice(s, e) for s, e in zip(start, end)]

        images_pred = images[start[0]:end[0], start[1]:end[1], start[2]:end[2], :]
        images_pred = images_pred.transpose(3, 0, 1, 2)

        seeds = self.seed[start[0]:end[0], start[1]:end[1], start[2]:end[2]].copy()
        init_prediction = np.isnan(seeds)
        seeds[init_prediction] = np.float32(logit(0.05))
        images_pred = torch.from_numpy(images_pred).float().unsqueeze(0)
        seeds = torch.from_numpy(seeds).float().unsqueeze(0).unsqueeze(0)

        # slice = seeds[:, :, seeds.shape[2] // 2, :, :].sigmoid()
        # seeds[:, :, seeds.shape[2] // 2, :, :] = slice

        input_data = torch.cat([images_pred, seeds], dim=1)
        input_data = Variable(input_data.cuda())

        logits = self.model(input_data)
        updated = (seeds.cuda() + logits).detach().cpu().numpy()
        # update_seed(updated, self.seed, self.model, pos)
        prob = expit(updated)

        return np.squeeze(prob), np.squeeze(updated)

    def update_at(self, pos):
        """Updates object mask prediction at a specific position.
        """
        global old_err
        off = self.input_size // 2  # zyx

        start = np.array(pos) - off
        # start_cent = np.array(pos) - 1

        end = start + self.input_size
        # end_cent = start_cent + 3

        # start_cent[0] = 0
        # end_cent[0] = 590

        sel = [slice(s, e) for s, e in zip(start, end)]

        logit_seed = np.array(self.seed[tuple(sel)])
        init_prediction = np.isnan(logit_seed)
        logit_seed[init_prediction] = np.float32(logit(0.05))

        prob_seed = expit(logit_seed)
        for _ in range(MAX_SELF_CONSISTENT_ITERS):
            """model inference"""
            prob, logits = self.predict(pos)
            break

        """update seed"""
        sel = [slice(s, e) for s, e in zip(start, end)]
        # sel_cent = [slice(s, e) for s, e in zip(start_cent, end_cent)]
        # Bias towards oversegmentation by making it impossible to reverse
        # disconnectedness predictions in the course of inference.
        th_max = logit(0.5)
        old_seed = self.seed[tuple(sel)]

        if np.mean(logits >= self.mov_thr) > 0:
            # Because (x > NaN) is always False, this mask excludes positions that
            # were previously uninitialized (i.e. set to NaN in old_seed).
            try:
                old_err = np.seterr(invalid='ignore')
                mask = ((old_seed < th_max) & (logits > old_seed))
            finally:
                np.seterr(**old_err)
            logits[mask] = old_seed[mask]

        # Update working space.
        self.seed[tuple(sel)] = logits

        return logits, sel

    def segment_at(self, start_pos, id, tag):
        t_lock = threading.Lock()

        # multi-threads
        global save_count
        global re_seged_count_mask
        global save_chunk
        global resume_seed
        global save_part

        try:
            if not self.is_valid_pos(start_pos, ignore_move_threshold=True):
                return


            if re_seged_count_mask[start_pos] >= self.re_seg_thr:
                print('skip', id)
                return


            self.seg_prob_i = np.zeros(self.shape, dtype=np.uint8)
            self.seed = np.zeros(self.shape, dtype=np.float32)
            self.init_seed(start_pos)
            self.reset_state(start_pos)

            if not self.movement_policy:
                # Add first element with arbitrary priority 1 (it will be consumed
                # right away anyway).
                item = (self.movement_policy.score_threshold * 2, start_pos)
                self.movement_policy.append(item)

            sel_i_s = [()]
            sel_cent_s = [()]
            step_iter = 0

            for pos in self.movement_policy:
                # Terminate early if the seed got too weak.
                # print(len(self.movement_policy.scored_coords))
                if self.seed[start_pos] < self.mov_thr:
                    break

                """stepping"""
                pred, sel_i = self.update_at(pos)
                step_iter += 1

                sel_i_s = sel_i  # updated_cube_fov_size

                """
                try:
                    mask = self.seed[tuple(sel_i_s)] >= self.seg_thr
                    self.seg_prob_i[tuple(sel_i_s)][mask] = quantize_probability(expit(self.seed[tuple(sel_i_s)][mask]))
                except RuntimeError:
                    return False
                # save the pred_mask out for each step
                skimage.io.imsave('./data/FFN_object1_inf_{}_step{}.tif'.format(id, num_iters), self.seg_prob_i)
                """

                """update_movable_location"""

                self.movement_policy.update(pred, pos)
                assert np.all(pred.shape == self.input_size)

            try:
                mask = (self.seed >= self.seg_thr)
                self.seg_prob_i[mask] = quantize_probability(expit(self.seed[mask]))
            except RuntimeError:
                return False

            mask_seg_prob_i = (self.seg_prob_i >= 128)
            if np.sum(mask_seg_prob_i) >= self.vox_thr:


                re_seged_count_mask[mask_seg_prob_i] += 1


                # save segmentation from each seed
                seg_prob_coords = np.argwhere(mask_seg_prob_i).astype('int16')

                t_lock.acquire()
                save_count += 1
                print('save_count', save_count, "process id", self.process_id)
                if save_count % save_chunk == 0:
                    save_part += 1
                id_save = '{}'.format(id)
                print("segmentation saved! seed:", id, "coord:", start_pos, "completed_num:", save_count)
                try:
                    with h5py.File(self.data_save_path + tag + "seg_of_seeds_test_part{}.h5".format(save_part), 'a') as f:
                        f.create_dataset(id_save, data=seg_prob_coords, compression='gzip')
                except OSError:
                    return True
                t_lock.release()


                return True
            else:
                print("too small", id)

        except RuntimeError:
            return False

