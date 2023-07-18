import albumentations as alb
import cv2
import json
import numpy as np
import os
import random
import re
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from skimage.transform import rotate as skimage_rotate
from segme.model.matting.fba_matting.distance import distance_transform
from segme.model.matting.fba_matting.twomap import twomap_transform
from segme.utils.albumentations import drop_unapplied
from segme.utils.matting import np as matting_np, tf as matting_tf
from segme.utils.common import rand_augment_matting

CROP_SIZE = 512
TOTAL_BOXES = 100
TRIMAP_SIZE = (3, 25)
UNKNOWN_LIMIT = (0., 1.)


def smart_crop(fg, alpha):
    nonzero = np.nonzero(alpha)
    top, bottom = nonzero[0].min(), nonzero[0].max()
    left, right = nonzero[1].min(), nonzero[1].max()

    fg_ = fg[top:bottom + 1, left:right + 1]
    alpha_ = alpha[top:bottom + 1, left:right + 1]

    cropped = top != 0, bottom != alpha.shape[0] - 1, left != 0, right != alpha.shape[1] - 1

    return fg_, alpha_, cropped


def smart_pad(fg, alpha, cropped):
    size = min(fg.shape[:2])
    size = max(size, CROP_SIZE) // 3

    top = size * int(cropped[0])
    bottom = size * int(cropped[1])
    left = size * int(cropped[2])
    right = size * int(cropped[3])

    fg_ = cv2.copyMakeBorder(fg, top, bottom, left, right, cv2.BORDER_CONSTANT)
    alpha_ = cv2.copyMakeBorder(alpha, top, bottom, left, right, cv2.BORDER_CONSTANT)

    return fg_, alpha_


def lanczos_upscale(fg, alpha, scale):
    alpha = np.squeeze(alpha)

    height = round(alpha.shape[0] * scale)
    width = round(alpha.shape[1] * scale)
    if alpha.shape[0] == height and alpha.shape[1] == width:
        return fg, alpha

    fg = Image.fromarray(fg)
    fg = fg.resize((width, height), resample=Image.Resampling.LANCZOS)
    fg = np.array(fg)

    alpha = Image.fromarray(alpha)
    alpha = alpha.resize((width, height), resample=Image.Resampling.LANCZOS)
    alpha = np.array(alpha)

    return fg, alpha


def prepare_train(fg, alpha):
    fg, alpha, cropped = smart_crop(fg, alpha)
    curr_actual = max(alpha.shape[:2])
    targ_actual = max(curr_actual, int(CROP_SIZE * .75))  # not less than 3/4 of crop size
    targ_actual = min(targ_actual, curr_actual * 2)  # but no more than x2 of original size
    targ_scale = targ_actual / curr_actual

    fg, alpha = smart_pad(fg, alpha, cropped)
    if min(alpha.shape[:2]) * targ_scale < CROP_SIZE:
        targ_scale = CROP_SIZE / min(alpha.shape[:2])

    fg = matting_np.solve_fg(fg, alpha)  # prevents artifacts
    fg, alpha = lanczos_upscale(fg, alpha, targ_scale)
    assert min(alpha.shape[:2]) >= CROP_SIZE

    return fg, alpha, targ_scale > 2.


def prepare_test(fg, alpha, bgs, trimaps):
    # crop to be divisible by 32
    hcrop, wcrop = alpha.shape[0] % 32, alpha.shape[1] % 32
    tcrop, bcrop = hcrop // 2, alpha.shape[0] - hcrop + hcrop // 2
    lcrop, rcrop = wcrop // 2, alpha.shape[1] - wcrop + wcrop // 2

    alpha = alpha[tcrop:bcrop, lcrop:rcrop]
    fg = fg[tcrop:bcrop, lcrop:rcrop]
    fg = matting_np.solve_fg(fg, alpha)

    bgs = [bg[tcrop:bcrop, lcrop:rcrop] for bg in bgs]
    trimaps = [trimap[tcrop:bcrop, lcrop:rcrop] for trimap in trimaps]

    return fg, alpha, bgs, trimaps


def max_angle(image):
    min_size = min(image.shape[:2])
    min_size = min(min_size, CROP_SIZE * 2 ** 0.5 - 1e-6)
    assert min_size >= CROP_SIZE

    discriminant = 2 * CROP_SIZE ** 2 - min_size ** 2
    assert discriminant > 0.

    # if 0. == discriminant:
    #     return np.arcsin(min_size * 0.5 / CROP_SIZE).item() * 180. / np.pi

    return np.arcsin((min_size - discriminant ** 0.5) * 0.5 / CROP_SIZE).item() * 180. / np.pi


def nonmax_suppression(boxes, threshold):
    if not len(boxes):
        return boxes

    boxes = boxes.astype('float')
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.arange(len(boxes))

    pick = []
    while len(idxs):
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate([[last], np.where(overlap > threshold)[0]]))

    return boxes[pick].astype('int')


def rotate_biquintic(fg, alpha, angle):
    alpha = np.squeeze(alpha)[..., None]
    full = np.ones_like(alpha)

    combo = np.concatenate([fg, alpha, full], axis=-1)
    combo = skimage_rotate(combo, angle, resize=False, preserve_range=True, mode='constant', cval=0, order=5)
    combo = np.round(combo).astype('uint8')

    fg, alpha, full = combo[..., :3], combo[..., 3:4], combo[..., 4:]
    assert not set(np.unique(full)) - {0, 1}

    return fg, alpha, full


def crop_boxes(alpha, full, num_boxes=TOTAL_BOXES):
    height, width = alpha.shape[:2]
    assert min(height, width) >= CROP_SIZE, alpha.shape[:2]

    trimap = (alpha > 0) & (alpha < 255)
    trimap = np.squeeze(trimap)
    assert trimap.sum()

    # crop center indices
    indices = np.stack(trimap.nonzero(), axis=-1)
    np.random.shuffle(indices)
    indices = indices[:100 * num_boxes]
    indices = np.minimum(indices, [[height - CROP_SIZE // 2, width - CROP_SIZE // 2]])
    indices = np.maximum(indices, CROP_SIZE // 2)

    # estimate crop expand ratio
    ratios = np.concatenate([
        indices,
        height - indices[:, :1],
        width - indices[:, 1:],
    ], axis=-1).min(axis=-1, keepdims=True) / CROP_SIZE
    ratios = np.random.uniform(0.5, ratios, ratios.shape)

    # estimate boxes
    boxes = np.concatenate([
        indices - ratios * CROP_SIZE,
        indices + ratios * CROP_SIZE,
    ], axis=-1).astype('int64')
    assert boxes.min() >= 0 and boxes[:, 0].max() < height and boxes[:, 1].max() < width

    # drop boxes with more then 90% overlapped
    np.random.shuffle(boxes)
    boxes = nonmax_suppression(boxes, 0.9)

    # drop boxes with holes (after augmentation)
    holes = np.apply_along_axis(lambda b: (full[b[0]:b[2], b[1]:b[3]] == 0).sum(), 1, boxes)
    boxes = np.delete(boxes, np.where(holes > 0)[0], axis=0)

    # drop overlapped
    thold = 0.8
    prev = np.empty((0, 4))
    np.random.shuffle(boxes)
    while len(boxes) > num_boxes:
        prev = boxes.copy()
        boxes = nonmax_suppression(boxes, thold)
        thold -= .1
    boxes = boxes if not len(prev) else prev

    return boxes[:num_boxes]


def scaled_crops(fg, alpha, num_crops=TOTAL_BOXES, repeats=5):
    maxangle = max_angle(fg)
    rotates = round(maxangle * (repeats - 1) / 45.)
    angles = np.random.uniform(-maxangle, maxangle, size=repeats).tolist()

    crops = []
    for i in range(repeats):
        if i < rotates:
            fg_, alpha_, full_ = rotate_biquintic(fg, alpha, angles[i])
        else:
            fg_, alpha_, full_ = fg, alpha, np.ones_like(alpha, 'uint8')

        boxes = crop_boxes(alpha_, full_, num_crops)
        if not len(boxes):
            continue

        for box in boxes:
            assert full_[box[0]:box[2], box[1]:box[3]].all()
            fg__ = fg_[box[0]:box[2], box[1]:box[3]]
            alpha__ = alpha_[box[0]:box[2], box[1]:box[3]]
            crops.append((fg__, alpha__))

    # repeat of not enough
    assert len(crops)
    if len(crops) < num_crops:
        crops *= int(num_crops / len(crops)) + 1

    random.shuffle(crops)
    crops = crops[:num_crops]
    assert len(crops) == num_crops

    return crops


def crop_augment(fg, alpha, replay=False):
    interpolations = np.random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4], size=2)
    fg = cv2.resize(fg, (CROP_SIZE, CROP_SIZE), interpolation=interpolations[0])
    alpha = cv2.resize(alpha, (CROP_SIZE, CROP_SIZE), interpolation=interpolations[1])

    compose_cls = alb.ReplayCompose if replay else alb.Compose
    aug = compose_cls([
        # Color
        alb.RandomGamma(p=0.5),  # reported as most useful
        alb.OneOf([
            alb.CLAHE(),
            # alb.OneOf([alb.ChannelDropout(fill_value=value) for value in range(256)]), # disable for matting
            # alb.ChannelShuffle(), # on-the-fly
            # alb.ColorJitter(), # on-the-fly
            # alb.OneOf([alb.Equalize(by_channels=value) for value in [True, False]]), # disable for matting
            alb.FancyPCA(),
            # alb.PixelDropout(), # disable for matting
            alb.RGBShift(),
            alb.RandomToneCurve(),
            alb.Sharpen(alpha=(0.1, 0.4), p=0.1),
            # alb.ToGray(p=0.1), # disable for matting
            # alb.ToSepia(p=0.05), # disable for matting
            alb.UnsharpMask(p=0.1)
        ], p=0.4),

        # Blur
        alb.OneOf([
            alb.Blur(blur_limit=(3, 4)),
            alb.GaussianBlur(blur_limit=(3, 5)),
            alb.MedianBlur(blur_limit=3),
            alb.MotionBlur(blur_limit=7),
            alb.GlassBlur(max_delta=2, iterations=1, p=0.1),
        ], p=0.2),

        # Noise
        alb.OneOf([
            alb.GaussNoise(var_limit=(10.0, 500.0)),
            alb.ISONoise(color_shift=(0.0, 0.1), intensity=(0.1, 0.7)),
            alb.OneOf([
                alb.MultiplicativeNoise(multiplier=(0.95, 1.05), per_channel=value) for value in [True, False]], p=0.2),
            # alb.ImageCompression(quality_lower=70, quality_upper=99), # on-the-fly
            # alb.Posterize(num_bits=(6, 8)), # disable for matting
        ], p=0.1),
    ])

    augmented = aug(image=fg, mask=alpha)
    assert (augmented['mask'] == alpha).all()

    if replay:
        print(drop_unapplied(augmented['replay']))

    return augmented['image'], alpha


class MattingDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.'
    }

    def __init__(self, *, source_dirs, background_dirs, data_dir, train_aug=1, test_re='-test-', config=None,
                 version=None):
        super().__init__(data_dir=data_dir, config=config, version=version)

        if isinstance(source_dirs, str):
            source_dirs = [source_dirs]
        if not isinstance(source_dirs, list):
            raise ValueError('A list expected for source directories')
        source_dirs = [os.fspath(s) for s in source_dirs]

        bad = [s for s in source_dirs if not os.path.isdir(s)]
        if bad:
            raise ValueError('Some of source directories do not exist: {}'.format(bad))

        self.source_dirs = source_dirs
        self.background_dirs = background_dirs
        self.train_aug = train_aug
        self.test_re = test_re

        self.similar = {}
        self.bg_files = []
        self.bg_index = 0

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description='Alpha matting dataset',
            features=tfds.features.FeaturesDict({
                'alpha': tfds.features.Image(shape=(None, None, 1), dtype=tf.uint8, encoding_format='png'),
                'foreground': tfds.features.Image(shape=(None, None, 3), dtype=tf.uint8, encoding_format='jpeg'),
                'background': tfds.features.Image(shape=(None, None, 3), dtype=tf.uint8, encoding_format='jpeg'),
                'trimap': tfds.features.Image(shape=(None, None, 1), dtype=tf.uint8, encoding_format='jpeg')
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            tfds.Split.TRAIN: self._generate_examples(True),
            tfds.Split.VALIDATION: self._generate_examples(False),
        }

    def _generate_examples(self, training):
        if training:
            bg_dirs = self.background_dirs
            if isinstance(bg_dirs, str):
                bg_dirs = [bg_dirs]
            if not isinstance(bg_dirs, list):
                raise ValueError('A list expected for background directories')

            bg_bad = [s for s in bg_dirs if not os.path.isdir(s)]
            if bg_bad:
                raise ValueError('Some of background directories do not exist: {}'.format(bg_bad))

            bg_dirs = [os.fspath(s) for s in bg_dirs]
            for d in bg_dirs:
                for root, _, files in os.walk(d):
                    for file in files:
                        if file[-4:] not in {'.jpg', 'jpeg', '.png'}:
                            continue
                        self.bg_files.append(os.path.join(root, file))

            if not self.bg_files:
                raise ValueError('No backgrounds found in '.format(bg_dirs))

            random.shuffle(self.bg_files)

        for alpha_file in self._iterate_source(training):
            for key, alpha, fg, bg, trimap in self._transform_example(alpha_file, training):
                alpha = np.squeeze(alpha)[..., None]

                yield key, {
                    'alpha': alpha,
                    'foreground': fg,
                    'background': bg,
                    'trimap': trimap
                }

    def _iterate_source(self, training):
        for source_dir in self.source_dirs:
            for dirpath, _, filenames in os.walk(source_dir):
                for file in filenames:
                    if not file.endswith('similar.json'):
                        continue

                    with open(os.path.join(dirpath, file), 'rt') as f:
                        self.similar.update(json.load(f))

        for source_dir in self.source_dirs:
            for dirpath, _, filenames in os.walk(source_dir):
                for file in filenames:
                    if training == bool(re.search(self.test_re, os.path.join('/', dirpath, file))):
                        continue

                    alpha_ext = '-alpha.png'
                    if not file.endswith(alpha_ext):
                        continue
                    alpha_path = os.path.join(dirpath, file)

                    yield alpha_path

    def _next_bg(self):
        selected = self.bg_files[self.bg_index]

        self.bg_index += 1
        if len(self.bg_files) == self.bg_index:
            random.shuffle(self.files)
            self.bg_index = 0

        bg = cv2.cvtColor(cv2.imread(selected), cv2.COLOR_BGR2RGB)

        interpolations = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
        if min(bg.shape[:2]) >= CROP_SIZE:
            max_scale = min(bg.shape[:2]) / CROP_SIZE
            curr_scale = np.random.uniform(low=1., high=max_scale, size=1).item()
            aug = alb.Compose([
                alb.RandomCrop(round(CROP_SIZE * curr_scale), round(CROP_SIZE * curr_scale), p=1),
                alb.OneOf([alb.SmallestMaxSize(CROP_SIZE, interpolation=value) for value in interpolations], p=1),
            ])
        else:
            aug = alb.Compose([
                alb.OneOf([alb.SmallestMaxSize(CROP_SIZE, interpolation=value) for value in interpolations], p=1),
                alb.RandomCrop(CROP_SIZE, CROP_SIZE, p=1),
            ])
        augmented = aug(image=bg)

        return augmented['image']

    def _transform_example(self, alpha_file, training):
        fg_file = alpha_file.replace('-alpha.png', '-fg.jpg')
        assert os.path.isfile(fg_file), fg_file

        fg = cv2.cvtColor(cv2.imread(fg_file), cv2.COLOR_BGR2RGB)
        alpha = cv2.imread(alpha_file, cv2.IMREAD_GRAYSCALE)[..., None]
        assert alpha.shape[:2] == fg.shape[:2], alpha_file

        if training:
            num_crops = TOTAL_BOXES
            for k, v in self.similar.items():
                if k in alpha_file:
                    num_crops = max(1, round(TOTAL_BOXES / v))

            for i in range(self.train_aug):
                crops = scaled_crops(fg, alpha, num_crops)

                for j, (fg_, alpha_) in enumerate(crops):
                    fg_, alpha_ = crop_augment(fg_, alpha_)
                    bg_ = self._next_bg()

                    trimap_ = np.zeros((1, 1, 1), dtype='uint8')
                    assert fg_.shape[:2] == alpha_.shape[:2] == bg_.shape[:2], (
                        fg_.shape[:2], alpha_.shape[:2], bg_.shape[:2])

                    yield '{}_{}_{}'.format(alpha_file, i, j), alpha_, fg_, bg_, trimap_
        else:
            for i in range(100):
                bg_file = alpha_file.replace('-alpha.png', '-{}_bg.jpg'.format(str(i).zfill(2)))
                trimap_file = alpha_file.replace('-alpha.png', '-{}_trimap.png'.format(str(i).zfill(2)))
                if not os.path.isfile(bg_file) or not os.path.isfile(trimap_file):
                    continue

                bg = cv2.cvtColor(cv2.imread(bg_file), cv2.COLOR_BGR2RGB)
                trimap = cv2.imread(trimap_file, cv2.IMREAD_GRAYSCALE)[..., None]
                assert alpha.shape[:2] == bg.shape[:2], alpha_file
                assert alpha.shape[:2] == trimap.shape[:2], alpha_file
                assert not len(set(np.unique(trimap)) - {0, 128, 255})

                yield '{}_{}'.format(alpha_file, i), alpha, fg, bg, trimap


@tf.function(jit_compile=False)
def _augment_examples(examples):
    alpha = examples['alpha']
    foreground = examples['foreground']
    background = examples['background']

    alpha.set_shape(alpha.shape[:1] + [CROP_SIZE, CROP_SIZE, 1])
    foreground.set_shape(foreground.shape[:1] + [CROP_SIZE, CROP_SIZE, 3])
    background.set_shape(background.shape[:1] + [CROP_SIZE, CROP_SIZE, 3])

    alpha = matting_tf.augment_alpha(alpha)
    foreground, alpha = matting_tf.random_compose(foreground, alpha, trim=(max(TRIMAP_SIZE), 0.95), solve=True)
    foreground, [alpha], _ = rand_augment_matting(foreground, [alpha], None)
    # foreground = matting_tf.solve_fg(foreground, alpha)
    background = tf.random.shuffle(background)
    trimap = matting_tf.alpha_trimap(alpha, size=TRIMAP_SIZE)
    trimap = matting_tf.augment_trimap(trimap)

    return {
        'alpha': alpha,
        'foreground': foreground,
        'background': background,
        'trimap': trimap,
    }


@tf.function(jit_compile=True)
def _prepare_examples_mf(examples):
    alpha = tf.cast(examples['alpha'], 'float32') / 255.
    foreground = tf.cast(examples['foreground'], 'float32') / 255.
    background = tf.cast(examples['background'], 'float32') / 255.

    image = foreground * alpha + background * (1. - alpha)
    image = tf.cast(tf.round(image * 255.), 'uint8')

    features = {'image': image, 'trimap': examples['trimap']}
    labels = tf.concat([alpha, foreground, background], axis=-1)
    weights = tf.cast(examples['trimap'] == 128, 'float32')

    return features, (alpha, labels, labels, labels), (weights, None, None, None)


@tf.function(jit_compile=False)
def _prepare_examples_fba(examples):
    alpha = tf.cast(examples['alpha'], 'float32') / 255.
    foreground = tf.cast(examples['foreground'], 'float32') / 255.
    background = tf.cast(examples['background'], 'float32') / 255.

    image = foreground * alpha + background * (1. - alpha)
    image = tf.cast(tf.round(image * 255.), 'uint8')

    twomap = twomap_transform(examples['trimap'])
    distance = distance_transform(examples['trimap'])

    features = {'image': image, 'twomap': twomap, 'distance': distance}

    alfgbg = tf.concat([alpha, foreground, background], axis=-1)
    labels = (alfgbg, alpha, foreground, background)

    weight = tf.cast(examples['trimap'] == 128, 'float32')
    weights = (None, weight, None, None)

    return features, labels, weights


@tf.function(jit_compile=True)
def _normalize_trimap(examples):
    trimap = examples['trimap']
    trimap = tf.cast(trimap // 86, 'int32') * 128
    trimap = tf.cast(tf.clip_by_value(trimap, 0, 255), 'uint8')
    examples['trimap'] = trimap

    return examples


def make_dataset(data_dir, split_name, out_mode, batch_size=1):
    train_split = tfds.Split.TRAIN == split_name

    builder = MattingDataset(source_dirs=[], background_dirs=[], data_dir=data_dir)
    builder.download_and_prepare()

    dataset = builder.as_dataset(split=split_name, batch_size=None, shuffle_files=train_split)

    if train_split:
        dataset = dataset \
            .shuffle(batch_size * 8) \
            .batch(max(32, batch_size * 4), drop_remainder=True) \
            .map(_augment_examples, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .unbatch() \
            .shuffle(batch_size * 8) \
            .batch(batch_size, drop_remainder=True)
    else:
        dataset = dataset \
            .batch(1) \
            .map(_normalize_trimap, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if 'fba' == out_mode:
        dataset = dataset.map(_prepare_examples_fba, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif 'mf' == out_mode:
        dataset = dataset.map(_prepare_examples_mf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        raise ValueError('Unknown mode')

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
