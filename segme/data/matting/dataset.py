import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import albumentations as alb
import cv2
import numpy as np
from keras.src import ops
from keras.src.utils.module_utils import tensorflow as tf
from PIL import Image

from segme.data.common.module import tensorflow_datasets as tfds
from segme.data.common.randaug import rand_augment_matting
from segme.data.matting.trimap import alpha_trimap
from segme.model.matting.fba_matting.distance import distance_transform
from segme.model.matting.fba_matting.twomap import twomap_transform
from segme.ops.image import convert_image_dtype
from segme.utils.matting.fg import solve_fg as solve_fg_np
from segme.utils.matting.trimap import alpha_trimap as alpha_trimap_np


def _prepare_train(fg, alpha, crop_size):
    fg_, alpha_, empty = _crop_nonempty(fg, alpha)
    fg_, alpha_ = _pad_resize(fg_, alpha_, empty, crop_size)
    assert min(fg_.shape[:2]) >= crop_size

    fg_ = solve_fg_np(fg_, alpha_)

    return fg_, alpha_


def _crop_nonempty(fg, alpha):
    nonzero = np.nonzero(alpha)
    top, bottom = nonzero[0].min(), nonzero[0].max()
    left, right = nonzero[1].min(), nonzero[1].max()

    fg_ = fg[top : bottom + 1, left : right + 1]
    alpha_ = alpha[top : bottom + 1, left : right + 1]

    empty = (
        top != 0,
        bottom != fg.shape[0] - 1,
        left != 0,
        right != fg.shape[1] - 1,
    )

    return fg_, alpha_, empty


def _pad_resize(fg, alpha, empty, crop_size):
    pad_size = max(crop_size // 2, max(fg.shape[:2]) // 5)
    top = pad_size * int(empty[0])
    bottom = pad_size * int(empty[1])
    left = pad_size * int(empty[2])
    right = pad_size * int(empty[3])

    fg_ = cv2.copyMakeBorder(fg, top, bottom, left, right, cv2.BORDER_CONSTANT)
    alpha_ = cv2.copyMakeBorder(
        alpha, top, bottom, left, right, cv2.BORDER_CONSTANT
    )

    if min(fg_.shape[:2]) < crop_size:
        up_scale = crop_size / min(fg_.shape[:2])
        fg_, alpha_ = _lanczos_upscale(fg_, alpha_, up_scale)

    return fg_, alpha_


def _lanczos_upscale(fg, alpha, scale):
    height = round(fg.shape[0] * scale)
    width = round(fg.shape[1] * scale)

    fg_ = Image.fromarray(fg)
    fg_ = fg_.resize((width, height), resample=Image.Resampling.LANCZOS)
    fg_ = np.array(fg_)

    alpha_ = Image.fromarray(np.squeeze(alpha))
    alpha_ = alpha_.resize((width, height), resample=Image.Resampling.LANCZOS)
    alpha_ = np.array(alpha_)

    return fg_, alpha_


def _prepare_test(fg, alpha, bgs, trimaps):
    # crop to be divisible by 32
    height_crop, width_crop = fg.shape[0] % 32, fg.shape[1] % 32
    top, bottom = height_crop // 2, fg.shape[0] - height_crop + height_crop // 2
    left, right = width_crop // 2, fg.shape[1] - width_crop + width_crop // 2

    alpha_ = alpha[top:bottom, left:right]
    assert 0 == alpha_.shape[0] % 32
    assert 0 == alpha_.shape[1] % 32

    fg_ = fg[top:bottom, left:right]
    fg_ = solve_fg_np(fg_, alpha_)

    bgs_ = [bg[top:bottom, left:right] for bg in bgs]
    trimaps_ = [trimap[top:bottom, left:right] for trimap in trimaps]

    return fg_, alpha_, bgs_, trimaps_


def _rotated_crops(fg, alpha, num_crops, crop_size):
    angles = np.arange(0, _max_angle(fg, crop_size) + 1, 8)
    angles *= np.sign(np.random.uniform(-1, 1, size=angles.size))
    angles = angles.tolist()

    crops = []
    for angle in angles:
        if 0.0 == angle:
            fg_, alpha_, full_ = fg, alpha, np.ones_like(alpha, "bool")
            max_boxes = num_crops * 2
        else:
            fg_, alpha_, full_ = _random_rotate(fg, alpha, angle)
            max_boxes = max(
                1, int(num_crops * min(1, min(fg.shape[:2]) / crop_size - 0.5))
            )

        boxes = _crop_boxes(alpha_, full_, max_boxes, crop_size)
        if not len(boxes):
            continue

        for box in boxes:
            assert (
                full_[box[0], box[1]]
                & full_[box[2] - 1, box[3] - 1]
                & full_[box[0], box[3] - 1]
                & full_[box[2] - 1, box[1]]
            )
            fg__ = fg_[box[0] : box[2], box[1] : box[3]]
            alpha__ = alpha_[box[0] : box[2], box[1] : box[3]]
            crops.append((fg__, alpha__))

    # repeat of not enough
    assert len(crops)
    if len(crops) < num_crops:
        crops *= int(num_crops / len(crops)) + 1
        assert len(crops) >= num_crops

    random.shuffle(crops)
    crops = crops[:num_crops]

    return crops


def _max_angle(image, crop_size):
    min_size = min(image.shape[:2])
    min_size = min(min_size, crop_size * 2**0.5 - 1e-6)
    assert min_size >= crop_size

    discriminant = 2 * crop_size**2 - min_size**2
    assert discriminant > 0.0

    # if 0. == discriminant:
    #     return np.arcsin(min_size * 0.5 / CROP_SIZE).item() * 180. / np.pi

    return (
        np.arcsin((min_size - discriminant**0.5) * 0.5 / crop_size)
        * 180.0
        / np.pi
    ).item()


def _random_rotate(fg, alpha, angle):
    alpha = np.squeeze(alpha)[..., None]
    full = np.ones_like(alpha) * 128

    interpolation = np.random.choice(
        [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    )

    size = np.array(fg.shape[1::-1], "float32")
    rotmat = cv2.getRotationMatrix2D(size / 2 - 0.5, angle, 1.0)
    dsize = np.abs(rotmat[:, :2]) @ size
    rotmat[:, -1] += (dsize - size) / 2.0 - 0.5

    fg = cv2.warpAffine(fg, rotmat, dsize.astype("int32"), flags=interpolation)
    alpha = cv2.warpAffine(
        alpha, rotmat, dsize.astype("int32"), flags=interpolation
    )
    full = cv2.warpAffine(
        full, rotmat, dsize.astype("int32"), flags=interpolation
    )
    full = (full == 128).astype("bool")

    return fg, alpha, full


def _crop_boxes(alpha, full, max_boxes, crop_size):
    alpha = np.squeeze(alpha)
    full = np.squeeze(full)

    height, width = alpha.shape[:2]
    assert min(height, width) >= crop_size

    unkmap = (alpha > 0) & (alpha < 255)
    assert unkmap.sum()

    # estimate centers
    indices = np.stack(unkmap.nonzero(), axis=-1)
    np.random.shuffle(indices)
    indices = indices[: max_boxes * 100]

    # estimate sizes
    ratios = np.random.uniform(
        0.5, min(height, width) / crop_size, indices.shape
    )
    deltas = (ratios * crop_size / 2).astype("int32")
    indices -= np.minimum(indices - deltas, 0)
    indices -= np.maximum(indices + deltas - [[height, width]] + 1, 0)

    boxes = np.concatenate(
        [
            # boxes with fake (minimal) scale for NMS
            indices - crop_size // 2 // 2,
            indices + crop_size // 2 // 2,
            # boxes with real scale for cropping
            indices - deltas,
            indices + deltas,
        ],
        axis=-1,
    )
    assert (
        boxes[4:].min() >= 0
        and boxes[:, 6].max() < height
        and boxes[:, 7].max() < width
    )

    # drop boxes with holes (after rotation)
    ground = (
        full[boxes[:, 4], boxes[:, 5]]
        & full[boxes[:, 6] - 1, boxes[:, 7] - 1]
        & full[boxes[:, 4], boxes[:, 7] - 1]
        & full[boxes[:, 6] - 1, boxes[:, 5]]
    )
    boxes = boxes[ground]

    # drop boxes with ~<4px diff (keep largest first)
    areas = (boxes[:, 6] - boxes[:, 4]) * (boxes[:, 7] - boxes[:, 5])
    boxes = boxes[np.argsort(areas)]
    boxes = _nonmax_suppression(boxes, 0.95)

    np.random.shuffle(boxes)
    boxes = boxes[:max_boxes, 4:]

    return boxes


def _nonmax_suppression(boxes, threshold):
    if len(boxes) < 2:
        return boxes

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    idxs = np.arange(len(boxes))

    pick = []
    while len(idxs):
        last = len(idxs) - 1
        idx = idxs[last]
        pick.append(idx)

        yy1 = np.maximum(y1[idx], y1[idxs[:last]])
        xx1 = np.maximum(x1[idx], x1[idxs[:last]])
        yy2 = np.minimum(y2[idx], y2[idxs[:last]])
        xx2 = np.minimum(x2[idx], x2[idxs[:last]])

        dh = np.maximum(yy2 - yy1, 0)
        dw = np.maximum(xx2 - xx1, 0)

        intersection = dw * dh
        union = areas[idxs[:last]] + areas[0] - intersection
        iou = intersection / union

        drop = np.where(iou > threshold)[0]
        drop = np.append(drop, len(idxs) - 1)
        idxs = np.delete(idxs, drop)

    return boxes[pick]


class MattingDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    def __init__(
        self,
        *,
        source_dirs,
        background_dirs,
        data_dir,
        crop_size,
        train_crops=100,
        train_aug=1,
        test_re="-test-",
        config=None,
        version=None,
    ):
        super().__init__(data_dir=data_dir, config=config, version=version)

        if isinstance(source_dirs, str):
            source_dirs = [source_dirs]
        if not isinstance(source_dirs, list):
            raise ValueError("A list expected for source directories")
        bad_source = [s for s in source_dirs if not os.path.isdir(s)]
        if bad_source:
            raise ValueError(
                f"Some of source directories do not exist: {bad_source}"
            )

        if isinstance(background_dirs, str):
            background_dirs = [background_dirs]
        if not isinstance(background_dirs, list):
            raise ValueError("A list expected for background directories")
        bad_background = [b for b in background_dirs if not os.path.isdir(b)]
        if bad_background:
            raise ValueError(
                f"Some of background directories do not exist: {bad_background}"
            )

        self.source_dirs = source_dirs
        self.background_dirs = background_dirs
        self.crop_size = crop_size
        self.train_crops = train_crops
        self.train_aug = train_aug
        self.test_re = test_re

        self.sim_stats = None
        self.bg_files = None
        self.alb_augs = None
        self.bg_index = 0
        self.bg_cache = []

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="Alpha matting dataset",
            features=tfds.features.FeaturesDict(
                {
                    "alpha": tfds.features.Image(
                        shape=(None, None, 1),
                        dtype=tf.uint8,
                        encoding_format="png",
                    ),
                    "foreground": tfds.features.Image(
                        shape=(None, None, 3),
                        dtype=tf.uint8,
                        encoding_format="png",
                    ),
                    "background": tfds.features.Image(
                        shape=(None, None, 3),
                        dtype=tf.uint8,
                        encoding_format="png",
                    ),
                    "trimap": tfds.features.Image(
                        shape=(None, None, 1),
                        dtype=tf.uint8,
                        encoding_format="png",
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            tfds.Split.TRAIN: self._generate_examples(tfds.Split.TRAIN),
            tfds.Split.VALIDATION: self._generate_examples(
                tfds.Split.VALIDATION
            ),
            tfds.Split.TEST: self._generate_examples(tfds.Split.TEST),
        }

    def _generate_examples(self, split):
        self._init_similar()
        self._init_backgrounds()
        self._init_albumetantions()

        for alpha_file in self._iterate_source(split):
            for key, alpha, fg, bg, trimap in self._transform_example(
                alpha_file, split
            ):
                alpha = np.squeeze(alpha)[..., None]

                yield (
                    key,
                    {
                        "alpha": alpha,
                        "foreground": fg,
                        "background": bg,
                        "trimap": trimap,
                    },
                )

    def _init_similar(self):
        if self.sim_stats is not None:
            return

        self.sim_stats = {}
        for source_dir in self.source_dirs:
            for dirpath, _, filenames in os.walk(source_dir):
                for file in filenames:
                    if not file.endswith("similar.json"):
                        continue

                    with open(os.path.join(dirpath, file), "rt") as f:
                        self.sim_stats.update(json.load(f))

    def _init_backgrounds(self):
        if self.bg_files is not None:
            return

        self.bg_files = []
        for background_dir in self.background_dirs:
            for root, _, files in os.walk(background_dir):
                for file in files:
                    if file[-4:] not in {".jpg", "jpeg", ".png"}:
                        continue
                    self.bg_files.append(os.path.join(root, file))

        if not self.bg_files:
            raise ValueError("No backgrounds found")

        random.shuffle(self.bg_files)

    def _init_albumetantions(self):
        if self.alb_augs is not None:
            return

        interpolations = [
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ]

        self.alb_augs = {
            "crop": alb.Compose(
                [
                    alb.OneOf(
                        [
                            alb.Resize(
                                self.crop_size,
                                self.crop_size,
                                interpolation=value,
                                mask_interpolation=value,
                            )
                            for value in interpolations
                        ],
                        p=1,
                    ),
                    # Color
                    alb.RandomGamma(p=0.5),  # reported as most useful
                    alb.OneOf(
                        [
                            alb.CLAHE(),
                            # Disable for matting
                            # alb.OneOf([
                            #   alb.ChannelDropout(fill_value=value)
                            #   for value in range(256)]),
                            # On-the-fly
                            # alb.ChannelShuffle(),
                            # Disable for matting
                            # alb.ChromaticAberration(mode="random"),
                            alb.ColorJitter(),
                            # Disable for matting
                            # alb.OneOf([
                            #   alb.Equalize(by_channels=value)
                            #   for value in [True, False]]),
                            alb.FancyPCA(),
                            alb.HueSaturationValue(),
                            # Disable for matting
                            # alb.PixelDropout(),
                            alb.OneOf(
                                [
                                    alb.PlanckianJitter("blackbody"),
                                    alb.PlanckianJitter("cied"),
                                ]
                            ),
                            alb.RGBShift(),
                            alb.RandomBrightnessContrast(),
                            alb.RandomGamma(),
                            alb.RandomToneCurve(),
                            alb.Sharpen(alpha=(0.1, 0.4), p=0.1),
                            # Disable for matting
                            # alb.ToGray(p=0.1),
                            # Disable for matting
                            # alb.ToSepia(p=0.05),
                            alb.UnsharpMask(p=0.1),
                        ],
                        p=0.4,
                    ),
                    # Blur
                    alb.OneOf(
                        [
                            alb.Blur(blur_limit=(3, 5)),
                            alb.Defocus(radius=(3, 7)),
                            alb.GaussianBlur(blur_limit=(3, 5)),
                            alb.MedianBlur(blur_limit=3),
                            alb.MotionBlur(blur_limit=(3, 9)),
                            alb.GlassBlur(max_delta=2, iterations=1, p=0.1),
                        ],
                        p=0.1,
                    ),
                    # Noise
                    alb.OneOf(
                        [
                            # Disable for matting
                            # alb.OneOf(
                            #     [
                            #         alb.Downscale(
                            #             scale_range=(0.75, 0.95),
                            #             interpolation_pair={
                            #                 "downscale": i1,
                            #                 "upscale": i2,
                            #             },
                            #         )
                            #         for i1 in INTERPOLATIONS
                            #         for i2 in INTERPOLATIONS
                            #     ]
                            # ),
                            alb.GaussNoise(std_range=(0.01, 0.1)),
                            alb.ISONoise(
                                color_shift=(0.0, 0.1), intensity=(0.1, 0.7)
                            ),
                            alb.OneOf(
                                [
                                    alb.MultiplicativeNoise(
                                        multiplier=(0.95, 1.05),
                                        per_channel=value,
                                    )
                                    for value in [True, False]
                                ],
                                p=0.2,
                            ),
                            # On-the-fly
                            # alb.ImageCompression(
                            #   quality_lower=70, quality_upper=99),
                            # Disable for matting
                            # alb.Posterize(num_bits=(6, 8)),
                        ],
                        p=0.1,
                    ),
                ]
            ),
            "back": alb.Compose(
                [
                    alb.OneOf(
                        [
                            alb.RandomSizedCrop(
                                (self.crop_size // 2, self.crop_size * 2),
                                (self.crop_size, self.crop_size),
                                interpolation=value,
                            )
                            # without cv2.INTER_LANCZOS4
                            for value in interpolations[:-1]
                        ],
                        p=1,
                    ),
                    alb.OneOf(
                        [
                            alb.Blur(blur_limit=(3, 35)),
                            alb.Defocus(radius=(3, 35)),
                            alb.GaussianBlur(blur_limit=(3, 35)),
                            # alb.MedianBlur(blur_limit=3),
                            alb.MotionBlur(blur_limit=(3, 35)),
                            alb.OneOf(
                                [
                                    alb.GlassBlur(max_delta=i, iterations=1)
                                    for i in range(1, 20)
                                ]
                            ),
                        ],
                        p=0.1,
                    ),
                ]
            ),
            "valid": alb.Compose(
                [
                    alb.SmallestMaxSize(
                        self.crop_size,
                        interpolation=cv2.INTER_LANCZOS4,
                        mask_interpolation=cv2.INTER_NEAREST,
                        p=1,
                    ),
                    alb.CenterCrop(self.crop_size, self.crop_size, p=1),
                ],
                additional_targets={"fg": "image", "bg": "image"},
            ),
        }

    def _iterate_source(self, split):
        for source_dir in self.source_dirs:
            for dirpath, _, filenames in os.walk(source_dir):
                for file in filenames:
                    if (tfds.Split.TRAIN == split) == bool(
                        re.search(
                            self.test_re, os.path.join("/", dirpath, file)
                        )
                    ):
                        continue

                    if not file.endswith("-alpha.png"):
                        continue

                    yield os.path.join(dirpath, file)

    def _transform_example(self, alpha_file, split):
        fg_file = alpha_file.replace("-alpha.png", "-fg.png")
        assert os.path.isfile(fg_file), fg_file

        fg = cv2.cvtColor(cv2.imread(fg_file), cv2.COLOR_BGR2RGB)
        alpha = cv2.imread(alpha_file, cv2.IMREAD_GRAYSCALE)[..., None]
        assert alpha.shape[:2] == fg.shape[:2], alpha_file

        if tfds.Split.TRAIN == split:
            num_crops = self.train_crops
            for k, v in self.sim_stats.items():
                if k in alpha_file:
                    num_crops = max(1, round(self.train_crops / v))

            for i in range(self.train_aug):
                crops = _rotated_crops(fg, alpha, num_crops, self.crop_size)
                crops = self._augment_crops(crops)

                for j, (fg_, alpha_, trimap_) in enumerate(crops):
                    bg_ = self._next_background()

                    assert alpha_.shape[:2] == fg_.shape[:2], alpha_file
                    assert alpha_.shape[:2] == bg_.shape[:2], alpha_file
                    assert alpha_.shape[:2] == trimap_.shape[:2], alpha_file
                    assert not len(set(np.unique(trimap_)) - {0, 128, 255})

                    yield (
                        "train_{}_{}_{}".format(alpha_file, i, j),
                        alpha_,
                        fg_,
                        bg_,
                        trimap_,
                    )
        elif tfds.Split.VALIDATION == split:
            for i in range(100):
                bg_file = alpha_file.replace(
                    "-alpha.png", "-{}_bg.png".format(str(i).zfill(2))
                )
                if not os.path.isfile(bg_file):
                    continue

                trimap_file = alpha_file.replace(
                    "-alpha.png", "-{}_trimap.png".format(str(i).zfill(2))
                )
                if not os.path.isfile(trimap_file):
                    continue

                bg = cv2.cvtColor(cv2.imread(bg_file), cv2.COLOR_BGR2RGB)
                trimap = cv2.imread(trimap_file, cv2.IMREAD_GRAYSCALE)[
                    ..., None
                ]
                alpha_, fg_, bg_, trimap_ = self._resize_valid(
                    alpha, fg, bg, trimap
                )

                assert alpha_.shape[:2] == fg_.shape[:2], alpha_file
                assert alpha_.shape[:2] == bg_.shape[:2], alpha_file
                assert alpha_.shape[:2] == trimap_.shape[:2], alpha_file
                assert not len(set(np.unique(trimap_)) - {0, 128, 255})

                yield (
                    "valid_{}_{}".format(alpha_file, i),
                    alpha_,
                    fg_,
                    bg_,
                    trimap_,
                )
        else:  # tfds.Split.TEST
            for i in range(100):
                bg_file = alpha_file.replace(
                    "-alpha.png", "-{}_bg.png".format(str(i).zfill(2))
                )
                if not os.path.isfile(bg_file):
                    continue

                trimap_file = alpha_file.replace(
                    "-alpha.png", "-{}_trimap.png".format(str(i).zfill(2))
                )
                if not os.path.isfile(trimap_file):
                    continue

                bg = cv2.cvtColor(cv2.imread(bg_file), cv2.COLOR_BGR2RGB)
                trimap = cv2.imread(trimap_file, cv2.IMREAD_GRAYSCALE)[
                    ..., None
                ]
                assert alpha.shape[:2] == bg.shape[:2], alpha_file
                assert alpha.shape[:2] == trimap.shape[:2], alpha_file
                assert not len(set(np.unique(trimap)) - {0, 128, 255})

                yield "test_{}_{}".format(alpha_file, i), alpha, fg, bg, trimap

    def _augment_crops(self, crops):
        with ThreadPoolExecutor() as executor:
            crops_ = list(executor.map(self._augment_crop, crops))

        return crops_

    def _augment_crop(self, crop):
        augmented = self.alb_augs["crop"](image=crop[0], mask=crop[1])
        fg_ = augmented["image"]
        alpha_ = augmented["mask"]
        fg_ = solve_fg_np(fg_, alpha_)
        trimap_ = alpha_trimap_np(alpha_, 3)
        trimap_ = np.squeeze(trimap_)[..., None]

        return fg_, alpha_, trimap_

    def _next_background(self):
        if not self.bg_cache:
            selected = self.bg_files[self.bg_index : self.bg_index + 64]
            self.bg_index += len(selected)
            if len(self.bg_files) == self.bg_index:
                random.shuffle(self.bg_files)
                self.bg_index = 0

            with ThreadPoolExecutor() as executor:
                self.bg_cache = list(
                    executor.map(self._load_background, selected)
                )
                self.bg_cache = [bg for bg in self.bg_cache if bg is not None]
                assert len(self.bg_cache)

        return self.bg_cache.pop(0)

    def _load_background(self, file):
        bg = None
        for _ in range(64):
            bg = cv2.imread(file)
            if bg is not None:
                bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
                bg = self.alb_augs["back"](image=bg)["image"]
                break

        return bg

    def _resize_valid(self, alpha, fg, bg, trimap):
        cropped = self.alb_augs["valid"](image=alpha, fg=fg, bg=bg, mask=trimap)
        alpha_ = cropped["image"]
        fg_ = cropped["fg"]
        bg_ = cropped["bg"]
        trimap_ = cropped["mask"]

        return alpha_, fg_, bg_, trimap_


def make_dataset(
    data_dir,
    crop_size,
    split_name,
    out_mode,
    batch_size=1,
    num_repeats=1,
    max_trimap=25,
):
    builder = MattingDataset(
        source_dirs=[],
        background_dirs=[],
        data_dir=data_dir,
        crop_size=crop_size,
    )
    builder.download_and_prepare()

    dataset = builder.as_dataset(
        split=split_name,
        batch_size=None,
        shuffle_files=tfds.Split.TRAIN == split_name,
    )

    if tfds.Split.TRAIN == split_name:
        augment_examples = partial(
            _augment_examples, crop_size=crop_size, max_trimap=max_trimap
        )
        dataset = (
            dataset.shuffle(batch_size * 8)
            .batch(max(32, batch_size * 4), drop_remainder=True)
            .map(
                augment_examples,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .unbatch()
            .shuffle(batch_size * 8)
            .batch(batch_size, drop_remainder=True)
        )
    elif tfds.Split.VALIDATION == split_name:
        prepare_crop = partial(_prepare_crop, crop_size=crop_size)
        dataset = dataset.batch(batch_size, drop_remainder=True).map(
            prepare_crop,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    else:
        dataset = dataset.batch(1).map(
            _prepare_trimap, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    if "com" == out_mode:
        dataset = dataset.map(
            _prepare_com,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    elif "fba" == out_mode:
        dataset = dataset.map(
            _prepare_fba,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    elif out_mode.startswith("exp") and out_mode[3:4].isnumeric():
        scales = int(out_mode[3:4])
        prepare_exp = partial(_prepare_exp, scales=scales)
        dataset = dataset.map(
            prepare_exp,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    else:
        raise ValueError("Unknown mode")

    if num_repeats > 1:
        dataset = dataset.repeat(num_repeats)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


@tf.function(jit_compile=False)
def _augment_examples(examples, crop_size, max_trimap):
    examples = _prepare_crop(examples, crop_size)

    alpha = examples["alpha"]
    foreground = examples["foreground"]
    background = examples["background"]
    trimap = examples["trimap"]

    # alpha = augment_alpha(alpha)
    foreground, [alpha, trimap], _ = rand_augment_matting(
        foreground, [alpha, trimap], None
    )
    # foreground = solve_fg(  # re-solve fg when alpha changed
    #   foreground, alpha, kappa=0.334, steps=3)  # for crop size 512
    background = ops.random.shuffle(background)
    if max_trimap > 3:
        trimap = alpha_trimap(trimap, size=[0, max_trimap - 3])
    # trimap = augment_trimap(trimap)

    return {
        "alpha": alpha,
        "foreground": foreground,
        "background": background,
        "trimap": trimap,
    }


@tf.function(jit_compile=True)
def _prepare_crop(examples, crop_size):
    examples = _prepare_trimap(examples)

    alpha = examples["alpha"]
    foreground = examples["foreground"]
    background = examples["background"]
    trimap = examples["trimap"]

    batch_size = alpha.shape[0]
    alpha.set_shape((batch_size, crop_size, crop_size, 1))
    foreground.set_shape((batch_size, crop_size, crop_size, 3))
    background.set_shape((batch_size, crop_size, crop_size, 3))
    trimap.set_shape((batch_size, crop_size, crop_size, 1))

    return {
        "alpha": alpha,
        "foreground": foreground,
        "background": background,
        "trimap": trimap,
    }


@tf.function(jit_compile=True)
def _prepare_trimap(examples):
    trimap = examples["trimap"]
    trimap = ops.where(trimap < 64, 0, trimap)
    trimap = ops.where((trimap >= 64) & (trimap < 192), 128, trimap)
    trimap = ops.where(trimap >= 192, 255, trimap)

    examples = {
        "alpha": examples["alpha"],
        "foreground": examples["foreground"],
        "background": examples["background"],
        "trimap": trimap,
    }

    return examples


@tf.function(jit_compile=True)
def _prepare_com(examples):
    alpha = convert_image_dtype(examples["alpha"], "float32")
    foreground = convert_image_dtype(examples["foreground"], "float32")
    background = convert_image_dtype(examples["background"], "float32")

    image = foreground * alpha + background * (1.0 - alpha)
    image = convert_image_dtype(image, "uint8")

    trimap = examples["trimap"]
    trimap_ = convert_image_dtype(trimap, "float32")

    features = {"image": image, "trimap": trimap}
    labels = ops.concatenate([alpha, foreground, background, trimap_], axis=-1)
    weights = ops.cast(trimap == 128, "float32")

    return (
        features,
        (labels, alpha, foreground, background),
        (None, weights, None, None),
    )


@tf.function(jit_compile=False)
def _prepare_fba(examples):
    features, labels, weights = _prepare_com(examples)

    twomap = twomap_transform(features["trimap"])
    distance = distance_transform(features["trimap"])
    features = {
        "image": features["features"],
        "twomap": twomap,
        "distance": distance,
    }

    return features, labels, weights


@tf.function(jit_compile=True)
def _prepare_exp(examples, scales):
    features, labels, weights = _prepare_com(examples)

    labels = labels[:1] * scales + labels[1:]
    weights = weights[:1] * scales + weights[1:]

    return features, labels, weights
