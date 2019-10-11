import itertools
import sys
from scipy.special import expit
from scipy.special import logit
import torch
import six
import numpy as np
from scipy import ndimage
import skimage
import skimage.feature
import logging
import weakref
from collections import namedtuple
from collections import deque
import time
from torch.autograd import Variable


MAX_SELF_CONSISTENT_ITERS = 32
HALT_SILENT = 0
PRINT_HALTS = 1
HALT_VERBOSE = 2

OriginInfo = namedtuple('OriginInfo', ['start_zyx', 'iters', 'walltime_sec'])
HaltInfo = namedtuple('HaltInfo', ['is_halt', 'extra_fetches'])


def make_seed(shape, pad=0.05, seed=0.95):
    """创建种子"""
    seed_array = np.full(list(shape), pad, dtype=np.float32)
    idx = tuple([slice(None)] + list(np.array(shape) // 2))
    seed_array[idx] = seed
    return seed_array


def fixed_offsets(seed, fov_moves, threshold=0.9):
    """offset偏移."""
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
    """根据中心坐标 crop patch"""
    target_shape = np.array(target_shape)

    start = coor - target_shape // 2
    end = start + target_shape

    assert np.all(start >= 0)

    selector = [slice(s, e) for s, e in zip(start, end)]
    cropped = data[tuple(selector)]

    if target_shape is not None:
        target_shape = np.array(target_shape)
        delta = target_shape - cropped.shape
        pre = delta // 2
        post = delta - delta // 2

        paddings = []  # no padding for batch
        paddings.extend(zip(pre, post))

        cropped = np.pad(cropped, paddings, mode='constant')

    return cropped


def crop_and_pad(data, offset, crop_shape, target_shape=None):
    """根据offset crop patch"""
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
        for iter, (image, targets, seed, coor) in enumerate(loader):
            for off in get_offsets(seed):
                predicted = crop_and_pad(seed, off, shape).unsqueeze(0)
                patches = crop_and_pad(image, off, shape).unsqueeze(0)
                labels = crop_and_pad(targets, off, shape).unsqueeze(0)
                offset = off

                yield predicted, patches, labels, offset


def get_batch(loader, batch_size, shape, get_offsets):
    def _batch(iterable):
        for batch_vals in iterable:
          yield zip(*batch_vals)

    for seeds, patches, labels, offsets in _batch(six.moves.zip(
        *[get_example(loader, shape, get_offsets) for _
            in range(batch_size)])):

        yield torch.cat(seeds, dim=0).float(), torch.cat(patches, dim=0).float(), \
              torch.cat(labels, dim=0).float(), offsets


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


class BaseSeedPolicy(object):
    """Base class for seed policies."""

    def __init__(self, canvas, **kwargs):
        """Initializes the policy.
        Args:
          canvas: inference Canvas object; simple policies use this to access
              basic geometry information such as the shape of the subvolume;
              more complex policies can access the raw image data, etc.
          **kwargs: other keyword arguments
        """
        del kwargs
        # TODO(mjanusz): Remove circular reference between Canvas and seed policies.
        self.canvas = weakref.proxy(canvas)
        self.coords = None
        self.idx = 0

        self._init_coords()

    def _init_coords(self):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self):
        """Returns the next seed point as (z, y, x).
        Does initial filtering of seed points to exclude locations that are
        too close to the image border.
        Returns:
          (z, y, x) tuples.
        Raises:
          StopIteration when the seeds are exhausted.
        """
        if self.coords is None:
            self._init_coords()

        while self.idx < self.coords.shape[0]:
            curr = self.coords[self.idx, :]
            self.idx += 1

            # TODO(mjanusz): Get rid of this.
            # Do early filtering of clearly invalid locations (too close to image
            # borders) as late filtering might be expensive.
            if (np.all(curr - self.canvas.margin >= 0) and
                np.all(curr + self.canvas.margin < self.canvas.shape)):
                yield tuple(curr)  # z, y, x

        raise StopIteration()

    def next(self):
        return self.__next__()

    def get_state(self):
        return self.coords, self.idx

    def set_state(self, state):
        self.coords, self.idx = state


def quantize_probability(prob):
    """Quantizes a probability map into a byte array."""
    ret = np.digitize(prob, np.linspace(0.0, 1.0, 255))

    # Digitize never uses the 0-th bucket.
    ret[np.isnan(prob)] = 0
    return ret.astype(np.uint8)


def get_scored_move_offsets(deltas, prob_map, threshold=0.9):
    """Looks for potential moves for a FFN.
    The possible moves are determined by extracting probability map values
    corresponding to cuboid faces at +/- deltas, and considering the highest
    probability value for every face.
    Args:
      deltas: (z,y,x) tuple of base move offsets for the 3 axes
      prob_map: current probability map as a (z,y,x) numpy array
      threshold: minimum score required at the new FoV center for a move to be
          considered valid
    Yields:
      tuples of:
        score (probability at the new FoV center),
        position offset tuple (z,y,x) relative to center of prob_map
      The order of the returned tuples is arbitrary and should not be depended
      upon. In particular, the tuples are not necessarily sorted by score.
    """
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


class PolicyPeaks(BaseSeedPolicy):
    """Attempts to find points away from edges in the image.
    Runs a 3d Sobel filter to detect edges in the raw data, followed
    by a distance transform and peak finding to identify seed points.
    """

    def _init_coords(self):
        logging.info('peaks: starting')

        # Edge detection.
        edges = ndimage.generic_gradient_magnitude(
            self.canvas.images.astype(np.float32),
            ndimage.sobel)

        # Adaptive thresholding.
        sigma = 49.0 / 6.0
        thresh_image = np.zeros(edges.shape, dtype=np.float32)
        ndimage.gaussian_filter(edges, sigma, output=thresh_image, mode='reflect')
        filt_edges = edges > thresh_image

        del edges, thresh_image

        # # This prevents a border effect where the large amount of masked area
        # # screws up the distance transform below.
        # if (self.canvas.restrictor is not None and
        #         self.canvas.restrictor.mask is not None):
        #     filt_edges[self.canvas.restrictor.mask] = 1

        logging.info('peaks: filtering done')
        dt = ndimage.distance_transform_edt(1 - filt_edges).astype(np.float32)
        logging.info('peaks: edt done')

        # Use a specifc seed for the noise so that results are reproducible
        # regardless of what happens before the policy is called.
        state = np.random.get_state()
        np.random.seed(42)
        idxs = skimage.feature.peak_local_max(
            dt + np.random.random(dt.shape) * 1e-4,
            indices=True, min_distance=3, threshold_abs=0, threshold_rel=0)
        np.random.set_state(state)

        # After skimage upgrade to 0.13.0 peak_local_max returns peaks in
        # descending order, versus ascending order previously.  Sort ascending to
        # maintain historic behavior.
        idxs = np.array(sorted((z, y, x) for z, y, x in idxs))

        logging.info('peaks: found %d local maxima', idxs.shape[0])
        self.coords = idxs


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

    def __init__(self, canvas, deltas=(4, 8, 8), score_threshold=0.9):
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
                                                threshold=self.score_threshold)
        scored_coords = sorted(scored_coords, reverse=True)
        for score, rel_coord in scored_coords:
              # convert to whole cube coordinates
              coord = [rel_coord[i] + position[i] for i in range(3)]
              self.scored_coords.append((score, coord))


class Canvas(object):

    def __init__(self, model, images, size, delta, seg_thr, mov_thr, act_thr):
        self.model = model
        self.images = images
        self.shape = images.shape
        self.input_size = np.array(size)
        self.margin = np.array(size) // 2
        self.seg_thr = logit(seg_thr)
        self.mov_thr = logit(mov_thr)
        self.act_thr = logit(act_thr)

        self.segmentation = np.zeros(images.shape, dtype=np.int32)
        self.seed = np.zeros(images.shape, dtype=np.float32)
        self.seg_prob = np.zeros(images.shape, dtype=np.uint8)

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
        if self.segmentation[pos] > 0:
            return False

        return True

    def predict(self, pos):
        """Runs a single step of FFN prediction.
        """
        # Top-left corner of the FoV.
        start = np.array(pos) - self.margin
        end = start + self.input_size

        assert np.all(start >= 0)

        # selector = [slice(s, e) for s, e in zip(start, end)]
        images = self.images[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        seeds = self.seed[start[0]:end[0], start[1]:end[1], start[2]:end[2]].copy()
        init_prediction = np.isnan(seeds)
        seeds[init_prediction] = np.float32(logit(0.05))
        images = torch.from_numpy(images).float().unsqueeze(0).unsqueeze(0)
        seeds = torch.from_numpy(seeds).float().unsqueeze(0).unsqueeze(0)

        slice = seeds[:, :, seeds.shape[2] // 2, :, :].sigmoid()
        seeds[:, :, seeds.shape[2] // 2, :, :] = slice

        input_data = torch.cat([images, seeds], dim=1)
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
        end = start + self.input_size
        sel = [slice(s, e) for s, e in zip(start, end)]
        logit_seed = np.array(self.seed[tuple(sel)])
        init_prediction = np.isnan(logit_seed)
        logit_seed[init_prediction] = np.float32(logit(0.05))

        prob_seed = expit(logit_seed)
        for _ in range(MAX_SELF_CONSISTENT_ITERS):
            prob, logits = self.predict(pos)
            break

            # diff = np.average(np.abs(prob_seed - prob))
            # if diff < self.options.consistency_threshold:
            #     break

            # prob_seed, logit_seed = prob, logits

        # if self.halt_signaler.is_halt(fetches=fetches, pos=pos,
        #                               orig_pos=start_pos,
        #                               counters=self.counters):
        #     logits[:] = np.float32(self.options.pad_value)

        sel = [slice(s, e) for s, e in zip(start, end)]

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

        return logits

    def segment_at(self, start_pos):
        self.init_seed(start_pos)
        num_iters = 0
        self.reset_state(start_pos)

        if not self.movement_policy:
            # Add first element with arbitrary priority 1 (it will be consumed
            # right away anyway).
            item = (self.movement_policy.score_threshold * 2, start_pos)
            self.movement_policy.append(item)
        for pos in self.movement_policy:
            # Terminate early if the seed got too weak.
            # print(len(self.movement_policy.scored_coords))
            if self.seed[start_pos] < self.mov_thr:
                  break

            # if not self.restrictor.is_valid_pos(pos):
            #     continue

            pred = self.update_at(pos)
            self._min_pos = np.minimum(self._min_pos, pos)
            self._max_pos = np.maximum(self._max_pos, pos)
            num_iters += 1

            self.movement_policy.update(pred, pos)

            assert np.all(pred.shape == self.input_size)

        return num_iters

    def segment_all(self):
        self.seed_policy = PolicyPeaks(self)
        mbd = np.array([1, 1, 1])
        iter = 0
        try:
            for pos in next(self.seed_policy):

                count = round(1.0 * iter / len(self.seed_policy.coords) * 50)

                if iter == 246:
                    print('done')

                sys.stdout.write('[ {}/{}: [{}{}]\r'.format(iter + 1, len(self.seed_policy.coords),
                                                            '#' * count, ' ' * (50 - count)))
                iter += 1

                if not self.is_valid_pos(pos, ignore_move_threshold=True):
                  continue

                low = np.array(pos) - mbd
                high = np.array(pos) + mbd + 1
                sel = [slice(s, e) for s, e in zip(low, high)]
                if np.any(self.segmentation[tuple(sel)] > 0):
                    self.segmentation[pos] = -1
                    continue

                seg_start = time.time()
                num_iters = self.segment_at(pos)
                t_seg = time.time() - seg_start

                if num_iters <= 0:
                    continue

                if self.seed[pos] < self.mov_thr:
                    # Mark this location as excluded.
                    if self.segmentation[pos] == 0:
                        self.segmentation[pos] = -1
                    continue

                sel = [slice(max(s, 0), e + 1) for s, e in zip(self._min_pos - self.input_size // 2, self._max_pos + self.input_size // 2)]
                mask = self.seed[tuple(sel)] >= self.seg_thr
                raw_segmented_voxels = np.sum(mask)
                overlapped_ids, counts = np.unique(self.segmentation[tuple(sel)][mask], return_counts=True)
                valid = overlapped_ids > 0
                overlapped_ids = overlapped_ids[valid]
                counts = counts[valid]
                mask &= self.segmentation[tuple(sel)] <= 0
                actual_segmented_voxels = np.sum(mask)
                if actual_segmented_voxels < 1000:
                    if self.segmentation[pos] == 0:
                        self.segmentation[pos] = -1
                    continue

                self.max_id += 1
                while self.max_id in self.origins:
                    self.max_id += 1

                self.segmentation[tuple(sel)][mask] = self.max_id
                self.seg_prob[tuple(sel)][mask] = quantize_probability(expit(self.seed[tuple(sel)][mask]))
                self.overlaps[self.max_id] = np.array([overlapped_ids, counts])
                self.origins[self.max_id] = OriginInfo(pos, num_iters, t_seg)

        except RuntimeError:
            return True
        
