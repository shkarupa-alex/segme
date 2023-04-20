import albumentations as alb
import cv2
import json
import numpy as np
import os
import random
import re
import tensorflow as tf
import tensorflow_datasets as tfds
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from skimage.transform import rotate as skimage_rotate
from segme.model.matting.fba_matting.distance import distance_transform
from segme.model.matting.fba_matting.twomap import twomap_transform
from segme.utils.albumentations import drop_unapplied
from segme.utils.matting import np as matting_np, tf as matting_tf
from segme.utils.common import rand_augment_safe

TRIMAP_SIZE = (3, 25)
CROP_SIZE = 512
TOTAL_BOXES = 50
MIN_TRIMAP = 0.05


def smart_pad(fg, alpha, size=CROP_SIZE):
    # TODO: add some context
    left = size * (1 - alpha[:, 0].any().astype('int32'))
    right = size * (1 - alpha[:, -1].any().astype('int32'))
    top = size * (1 - alpha[0].any().astype('int32'))
    bottom = size * (1 - alpha[-1].any().astype('int32'))

    fg_ = cv2.copyMakeBorder(fg, top, bottom, left, right, cv2.BORDER_REPLICATE)
    alpha_ = cv2.copyMakeBorder(alpha, top, bottom, left, right, cv2.BORDER_CONSTANT)

    return fg_, alpha_


def scale_augment(fg, alpha, scale):
    def _resize(src, size):
        src_ = Image.fromarray(src)
        assert src_.mode in {'RGB', 'L'}

        dst_ = src_.resize(size, resample=Image.Resampling.LANCZOS)
        dst = np.array(dst_)

        return dst

    width_height = round(alpha.shape[1] * scale), round(alpha.shape[0] * scale)

    fg = _resize(fg, width_height)
    alpha = _resize(alpha if 2 == len(alpha.shape) else alpha[..., 0], width_height)

    return fg, alpha


def rotate_augment(fg, alpha):
    def _rotate(src, ang):
        if 0 == angle:
            return src

        dst_ = skimage_rotate(src, ang, resize=False, preserve_range=True, mode='constant', cval=0, order=5)
        dst = np.round(dst_).astype('uint8')

        return dst

    angle = np.random.randint(-45, 45, size=1).item()
    fg = _rotate(fg, angle)
    alpha = _rotate(alpha, angle)
    full = _rotate(np.ones_like(alpha, 'uint8'), angle)

    assert not set(np.unique(full)) - {0, 1}

    return fg, alpha, full


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


def crop_boxes(alpha, full, num_boxes=TOTAL_BOXES):
    height, width = alpha.shape[:2]
    assert min(height, width) >= CROP_SIZE, alpha.shape[:2]

    trimap = matting_np.alpha_trimap(alpha, 3) == 128
    # trimap = (alpha > 0) & (alpha < 255)
    if 3 == len(trimap.shape):
        assert 1 == trimap.shape[-1]
        trimap = trimap[..., 0]
    assert trimap.sum()

    # crop center indices
    indices = np.stack(trimap.nonzero(), axis=-1)
    if indices.shape[0] > 2 * 100 * num_boxes:
        np.random.shuffle(indices)
        indices = indices[::indices.shape[0] // (100 * num_boxes)]
    indices = np.minimum(indices, [[height - CROP_SIZE // 2, width - CROP_SIZE // 2]])
    indices = np.maximum(indices, CROP_SIZE // 2)

    # estimate crop expand ratio [1.; 2.]
    ratios = np.concatenate([
        indices,
        height - indices[:, :1],
        width - indices[:, 1:],
    ], axis=-1).min(axis=-1, keepdims=True)
    ratios = np.minimum(ratios / CROP_SIZE, 2.)
    ratios = np.random.uniform(1., ratios, ratios.shape)

    # estimate boxes
    boxes = np.concatenate([
        indices - ratios * CROP_SIZE // 2,
        indices + ratios * CROP_SIZE // 2,
    ], axis=-1).astype('int64')

    # drop boxes with more then 90% overlapped
    np.random.shuffle(boxes)
    boxes = nonmax_suppression(boxes, 0.9)

    # drop boxes with holes (after augmentation)
    holes = np.apply_along_axis(lambda b: (full[b[0]:b[2], b[1]:b[3]] == 0).sum(), 1, boxes)
    boxes = np.delete(boxes, np.where(holes > 0)[0], axis=0)

    # drop overlapped
    thold = .8
    prev = np.empty((0, 4))
    np.random.shuffle(boxes)
    while len(boxes) > num_boxes:
        prev = boxes.copy()
        boxes = nonmax_suppression(boxes, thold)
        thold -= .1
    boxes = boxes if not len(prev) else prev

    return boxes[:num_boxes]


def scaled_crops(fg, alpha, num_crops=TOTAL_BOXES, repeats=5):
    actual_mask = alpha > 0
    height_index = actual_mask.any(1).nonzero()[0]
    actual_height = height_index.max() + 1 - height_index.min()
    width_index = actual_mask.any(0).nonzero()[0]
    actual_width = width_index.max() + 1 - width_index.min()
    curr_actual = max(actual_height, actual_width)

    # TODO: try all scales from actual size to CROP_SIZE
    targ_actual = max(curr_actual, int(CROP_SIZE * .75))  # not less then 3/4 of crop size
    targ_actual = min(targ_actual, curr_actual * 2)  # but no more then 2x of original size
    targ_actual = min(targ_actual, 2100)  # and no more then max test size

    targ_scale = targ_actual / curr_actual
    if min(alpha.shape[:2]) < CROP_SIZE:
        targ_scale = CROP_SIZE / min(alpha.shape[:2])

    fg, alpha = scale_augment(fg, alpha, targ_scale)

    crops = []
    for i in range(repeats):
        if 0 == i:
            fg_, alpha_, full_ = fg, alpha, np.ones_like(alpha, 'uint8')
        else:
            fg_, alpha_, full_ = rotate_augment(fg, alpha)

        boxes = crop_boxes(alpha_, full_, num_crops)
        if not len(boxes):
            continue

        for box in boxes:
            assert np.all(full_[box[0]:box[2], box[1]:box[3], ...] == 1)
            fg__ = fg_[box[0]:box[2], box[1]:box[3], ...]
            alpha__ = alpha_[box[0]:box[2], box[1]:box[3], ...]

            crops.append((fg__, alpha__))

    # repeat of not enough
    assert len(crops)
    if len(crops) < repeats * num_crops:
        crops_ = []
        for _ in range(int(repeats * num_crops / len(crops)) + 1):
            crops_.extend(crops)
        crops = crops_[:repeats * num_crops]
    assert len(crops) == repeats * num_crops

    return crops


class BackgroundGenerator:
    files = []
    index = 0

    def __init__(self, dirs):
        if isinstance(dirs, str):
            dirs = [dirs]
        if not isinstance(dirs, list):
            raise ValueError('A list expected for background directories')

        bad = [s for s in dirs if not os.path.isdir(s)]
        if bad:
            raise ValueError('Some of background directories do not exist: {}'.format(bad))

        dirs = [os.fspath(s) for s in dirs]

        for d in dirs:
            for root, _, files in os.walk(d):
                for file in files:
                    if file[-4:] not in {'.jpg', 'jpeg', '.png'}:
                        continue
                    self.files.append(os.path.join(root, file))

        if not self.files:
            raise ValueError('No backgrounds found in '.format(dirs))

        random.shuffle(self.files)

    def get(self, total):
        selected = []
        for _ in range(total):
            selected.append(self.files[self.index])
            self.index += 1

            if len(self.files) == self.index:
                random.shuffle(self.files)
                self.index = 0

        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = [pool.submit(self.crop, file=s) for s in selected]
            results = [r.result() for r in as_completed(futures)]

        for i in range(total):
            if results[i].shape[0] < CROP_SIZE or results[i].shape[1] < CROP_SIZE:
                raise ValueError('Wrong background shape after crop')

        return results

    def crop(self, file):
        bg = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)

        interpolations = [
            cv2.INTER_NEAREST_EXACT, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
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


def crop_augment(fg, alpha, replay=False):
    compose_cls = alb.ReplayCompose if replay else alb.Compose

    aug = compose_cls([
        # Color
        alb.RandomGamma(gamma_limit=(65, 160), p=0.7),  # reported as most useful
        alb.OneOf([
            alb.CLAHE(),
            # alb.OneOf([ # disable for matting
            #     alb.ChannelDropout(fill_value=value)
            #     for value in range(256)
            # ]),
            # alb.ChannelShuffle(), # on-the-fly
            alb.ColorJitter(),
            alb.OneOf([alb.Equalize(by_channels=value) for value in [True, False]]),
            alb.FancyPCA(),
            # alb.HueSaturationValue(), # on-the-fly
            # alb.PixelDropout(), # disable for matting
            alb.RGBShift(),
            # alb.RandomBrightnessContrast(), # on-the-fly
            alb.RandomToneCurve(),
            alb.Sharpen(),
            alb.ToGray(p=0.1),
            alb.ToSepia(p=0.05),
            alb.UnsharpMask()
        ], p=0.2),

        # Blur
        alb.OneOf([
            alb.Blur(blur_limit=(3, 4)),
            alb.GaussianBlur(blur_limit=(3, 5)),
            alb.MedianBlur(blur_limit=3),
            alb.MotionBlur(blur_limit=(3, 10)),
            alb.GlassBlur(max_delta=2, iterations=1, p=0.1),
        ], p=0.2),

        # Noise
        alb.OneOf([
            alb.GaussNoise(var_limit=(10.0, 500.0)),
            alb.ISONoise(color_shift=(0.0, 0.1), intensity=(0.1, 0.7)),
            alb.OneOf([
                alb.MultiplicativeNoise(multiplier=(0.9, 1.2), per_channel=value1, elementwise=value2)
                for value1, value2 in [(True, True), (True, False), (False, True), (False, False)]
            ]),
            alb.ImageCompression(quality_lower=70, quality_upper=99),
            alb.Posterize(num_bits=(6, 8)),
        ], p=0.3),
    ])

    augmented = aug(image=fg, mask=alpha)
    if replay:
        print(drop_unapplied(augmented['replay']))

    assert np.all(augmented['mask'] == alpha)
    fg = augmented['image']

    interpolations = np.random.choice([
        cv2.INTER_NEAREST_EXACT, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4], size=3)
    fg = cv2.resize(fg, (CROP_SIZE, CROP_SIZE), interpolation=interpolations[0])
    alpha = cv2.resize(alpha, (CROP_SIZE, CROP_SIZE), interpolation=interpolations[2])

    return fg, alpha


class MattingDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.'
    }

    def __init__(self, *, source_dirs, backgr_dirs, data_dir, train_aug=1, test_re='-test-', config=None, version=None):
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
        self.train_aug = train_aug
        self.test_re = test_re

        self.similar = {}

        self.bgen = None
        if backgr_dirs:
            self.bgen = BackgroundGenerator(backgr_dirs)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description='Alpha matting dataset',
            features=tfds.features.FeaturesDict({
                'alpha': tfds.features.Image(shape=(None, None, 1), dtype=tf.uint8, encoding_format='png'),
                'foreground': tfds.features.Image(shape=(None, None, 3), dtype=tf.uint8, encoding_format='jpeg'),
                'background': tfds.features.Image(shape=(None, None, 3), dtype=tf.uint8, encoding_format='jpeg'),
                'trimap': tfds.features.Image(shape=(None, None, 1), dtype=tf.uint8, encoding_format='png')
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            'train': self._generate_examples(True),
            'test': self._generate_examples(False),
        }

    def _generate_examples(self, training):
        for alpha_file in self._iterate_source(training):
            for key, alpha, fg, bg, trimap in self._transform_example(alpha_file, training):
                if 2 == len(alpha.shape):
                    alpha = alpha[..., None]

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

    def _transform_example(self, alpha_file, training):
        fg_file = alpha_file.replace('-alpha.png', '-solved.jpg')
        assert os.path.isfile(fg_file), fg_file

        fg = cv2.cvtColor(cv2.imread(fg_file), cv2.COLOR_BGR2RGB)
        alpha = cv2.imread(alpha_file, cv2.IMREAD_GRAYSCALE)[..., None]

        # if training:
        #     _, alpha = smart_pad(fg, alpha)

        assert alpha.shape[:2] == fg.shape[:2], alpha_file

        if training:
            num_crops = TOTAL_BOXES
            for k, v in self.similar.items():
                if k in alpha_file:
                    num_crops = max(1, round(TOTAL_BOXES / v))

            for i in range(self.train_aug):
                crops = scaled_crops(fg, alpha, num_crops)
                bgs = self.bgen.get(len(crops))

                for j, (fg_, alpha_) in enumerate(crops):
                    fg__, alpha__ = crop_augment(fg_, alpha_)
                    fg__ = matting_np.solve_fg(fg__, alpha__)
                    bg__ = bgs[j]

                    trimap__ = np.zeros((1, 1, 1), dtype='uint8')
                    assert fg__.shape[:2] == alpha__.shape[:2] == bg__.shape[:2], (
                        fg__.shape[:2], alpha__.shape[:2], bg__.shape[:2])

                    yield '{}_{}_{}'.format(alpha_file, i, j), alpha__, fg__, bg__, trimap__
        else:
            # crop to be divisible by 32
            hpad, wpad = alpha.shape[0] % 32, alpha.shape[1] % 32
            tpad, bpad = hpad // 2, alpha.shape[0] - hpad + hpad // 2
            lpad, rpad = wpad // 2, alpha.shape[1] - wpad + wpad // 2

            alpha_ = alpha[tpad:bpad, lpad:rpad]
            fg_ = fg[tpad:bpad, lpad:rpad]

            for i in range(100):
                bg_file = alpha_file.replace('-alpha.png', '-{}_bg.jpg'.format(str(i).zfill(2)))
                trimap_file = alpha_file.replace('-alpha.png', '-{}_trimap.png'.format(str(i).zfill(2)))
                if not os.path.isfile(bg_file) or not os.path.isfile(trimap_file):
                    continue

                bg = cv2.cvtColor(cv2.imread(bg_file), cv2.COLOR_BGR2RGB)
                trimap = cv2.imread(trimap_file, cv2.IMREAD_GRAYSCALE)[..., None]
                assert not len(set(np.unique(trimap)) - {0, 128, 255})

                bg_ = bg[tpad:bpad, lpad:rpad]
                trimap_ = trimap[tpad:bpad, lpad:rpad]

                yield '{}_{}'.format(alpha_file, i), alpha_, fg_, bg_, trimap_


@tf.function
def _augment_examples(examples, tri_bord, min_unk):
    alpha = examples['alpha']
    foreground = examples['foreground']
    background = examples['background']

    # alpha = matting_tf.augment_alpha(alpha)
    foreground, [alpha], _ = rand_augment_safe(foreground, [alpha], None)  # TODO
    background = tf.random.shuffle(background)
    # foreground, alpha, [background] = matting_tf.compose_two(foreground, alpha, [background], solve=True, prob=2/3)

    trimap = matting_tf.alpha_trimap(alpha, size=tri_bord)
    # TODO: random trimap erosion like in hqs_crm

    # unk_frac = tf.cast(trimap == 128, 'float32')
    unk_frac = tf.ones_like(trimap, 'float32') * 0.5
    unk_frac = tf.reduce_mean(unk_frac, axis=[1, 2, 3])
    accept = (unk_frac > min_unk) & (unk_frac < 1. - min_unk)

    # Augment trimap after acceptance estimation
    # trimap = matting_tf.augment_trimap(trimap)

    return {
        'alpha': alpha[accept],
        'foreground': foreground[accept],
        'background': background[accept],
        'trimap': trimap[accept]
    }


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

    weight_unknown = tf.cast(examples['trimap'] == 128, 'float32')
    weight_foreground = tf.cast(examples['trimap'] > 0, 'float32')
    weight_background = tf.cast(examples['trimap'] < 255, 'float32')

    # weight_size = tf.cast(tf.size(examples['trimap']), 'float32')
    # unknown_sum = tf.reduce_sum(weight_unknown, axis=[1, 2, 3], keepdims=True)
    # foreground_sum = tf.reduce_sum(weight_foreground, axis=[1, 2, 3], keepdims=True)
    # background_sum = tf.reduce_sum(weight_background, axis=[1, 2, 3], keepdims=True)
    # weight_alfgbg = tf.concat([
    #     weight_alpha * weight_size / alpha_sum,
    #     weight_foreground * weight_size / foreground_sum,
    #     weight_background * weight_size / background_sum
    # ], axis=-1)
    # weights = (weight_alfgbg, weight_alpha, None, None)

    # weight_unfgbg = tf.concat([weight_unknown,  weight_foreground, weight_background], axis=-1)
    # weights = (weight_unfgbg, weight_alpha, None, None)

    weights = (None, weight_unknown, None, None)

    return features, labels, weights


def make_dataset(data_dir, phase, mode, batch):
    builder = MattingDataset(source_dirs=[], backgr_dirs=[], data_dir=data_dir)
    builder.download_and_prepare()

    dataset = builder.as_dataset(split=phase, batch_size=None, shuffle_files=True)

    if 'train' == phase:
        dataset = dataset.shuffle(256)
        dataset = dataset.batch(32)
        dataset = dataset.map(lambda examples: _augment_examples(examples, TRIMAP_SIZE, MIN_TRIMAP),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.batch(1)

    if 'fba' == mode:
        dataset = dataset.map(_prepare_examples_fba, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif 'mf' == mode:
        dataset = dataset.map(_prepare_examples_mf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        raise ValueError('Unknown mode')

    if 'train' == phase:
        dataset = dataset.rebatch(batch)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
