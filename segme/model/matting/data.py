import json
import os
import random
import re

import albumentations as alb
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from albumentations.augmentations.geometric.functional import (
    rotate as alb_rotate,
)
from keras.src import ops
from PIL import Image

from segme.model.matting.fba_matting.distance import distance_transform
from segme.model.matting.fba_matting.twomap import twomap_transform
from segme.utils import matting
from segme.utils import matting_np
from segme.utils.albumentations import drop_unapplied
from segme.utils.common import rand_augment_matting

CROP_SIZE = 512
TOTAL_BOXES = 100
TRIMAP_SIZE = (3, 25)


def smart_crop(fg, alpha):
    nonzero = np.nonzero(alpha)
    top, bottom = nonzero[0].min(), nonzero[0].max()
    left, right = nonzero[1].min(), nonzero[1].max()

    fg_ = fg[top : bottom + 1, left : right + 1]
    alpha_ = alpha[top : bottom + 1, left : right + 1]

    cropped = (
        top != 0,
        bottom != alpha.shape[0] - 1,
        left != 0,
        right != alpha.shape[1] - 1,
    )

    return fg_, alpha_, cropped


def smart_pad(fg, alpha, cropped):
    size = min(fg.shape[:2])
    size = max(size, CROP_SIZE) // 3

    top = size * int(cropped[0])
    bottom = size * int(cropped[1])
    left = size * int(cropped[2])
    right = size * int(cropped[3])

    fg_ = cv2.copyMakeBorder(fg, top, bottom, left, right, cv2.BORDER_CONSTANT)
    alpha_ = cv2.copyMakeBorder(
        alpha, top, bottom, left, right, cv2.BORDER_CONSTANT
    )

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
    # TODO do not crop if then pad?
    fg, alpha, cropped = smart_crop(fg, alpha)
    curr_actual = max(alpha.shape[:2])
    targ_actual = max(
        curr_actual, int(CROP_SIZE * 0.75)
    )  # not less than 3/4 of crop size
    targ_actual = min(
        targ_actual, curr_actual * 2
    )  # but no more than x2 of original size
    targ_scale = targ_actual / curr_actual

    fg, alpha = smart_pad(fg, alpha, cropped)
    if min(alpha.shape[:2]) * targ_scale < CROP_SIZE:
        targ_scale = CROP_SIZE / min(alpha.shape[:2])

    fg = matting_np.solve_fg(fg, alpha)  # prevents artifacts
    fg, alpha = lanczos_upscale(fg, alpha, targ_scale)
    assert min(alpha.shape[:2]) >= CROP_SIZE

    return fg, alpha, targ_scale > 2.0


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
    # TODO: check max value
    min_size = min(image.shape[:2])
    min_size = min(min_size, CROP_SIZE * 2**0.5 - 1e-6)
    assert min_size >= CROP_SIZE

    discriminant = 2 * CROP_SIZE**2 - min_size**2
    assert discriminant > 0.0

    # if 0. == discriminant:
    #     return np.arcsin(min_size * 0.5 / CROP_SIZE).item() * 180. / np.pi

    return (
        np.arcsin((min_size - discriminant**0.5) * 0.5 / CROP_SIZE).item()
        * 180.0
        / np.pi
    )


def nonmax_suppression(boxes, threshold):
    if not len(boxes):
        return boxes

    boxes = boxes.astype("float")
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1) * (y2 - y1)
    idxs = np.arange(len(boxes))

    pick = []
    while len(idxs):
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])

        dh = np.maximum(yy2 - yy1, 0)
        dw = np.maximum(xx2 - xx1, 0)

        overlap = dw * dh
        iou = 2 * overlap / (area[idxs[:last]] + area[idxs[last]] - overlap)

        drop = np.where(iou > threshold)[0]
        idxs = np.delete(idxs, np.concatenate([[last], drop]))

    return boxes[pick].astype("int")


def random_rotate(fg, alpha, angle):
    alpha = np.squeeze(alpha)[..., None]
    full = np.ones_like(alpha) * 128

    interpolation = np.random.choice(
        [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    ).item()

    fga = np.concatenate([fg, alpha], axis=-1)
    # TODO: skimage_rotate(
    #   src, ang, resize=False, preserve_range=True, mode='constant',
    #   cval=0, order=5) ?
    fga = alb_rotate(
        fga,
        angle=angle,
        interpolation=interpolation,
        border_mode=cv2.BORDER_REFLECT_101,
    )
    full = alb_rotate(
        full,
        angle=angle,
        interpolation=interpolation,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
    )

    fg, alpha = fga[..., :3], fga[..., 3:]
    full = (full == 128).astype("bool")

    return fg, alpha, full


def crop_boxes(alpha, full, num_boxes=TOTAL_BOXES):
    alpha = np.squeeze(alpha)
    full = np.squeeze(full)

    height, width = alpha.shape[:2]
    assert min(height, width) >= CROP_SIZE

    trimap = matting_np.alpha_trimap(alpha, 3) == 128
    # trimap = (alpha > 0) & (alpha < 255)
    assert trimap.sum()

    # estimate boxes
    indices = np.stack(trimap.nonzero(), axis=-1)

    ratios = np.random.uniform(
        0.5, min(height, width) / CROP_SIZE, indices.shape
    )
    deltas = (ratios * CROP_SIZE / 2).astype("int64")

    indices -= np.minimum(indices - deltas, 0)
    indices -= np.maximum(indices + deltas - [[height, width]] + 1, 0)
    indices, unkidx = np.unique(indices, return_index=True, axis=0)
    deltas = deltas[unkidx]

    boxes = np.concatenate(
        [
            # boxes with fake (minimal) scale for NMS
            indices - CROP_SIZE // 2 // 2,
            indices + CROP_SIZE // 2 // 2,
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

    # TODO: drop fully transparent?

    # drop boxes with more then 99->98->...->1% overlapped
    thold = 0.99
    prev = np.empty((0, 4))
    np.random.shuffle(boxes)
    while len(boxes) > num_boxes and thold > 0.0:
        prev = boxes.copy()
        boxes = nonmax_suppression(boxes, thold)
        thold -= 0.01
    boxes = boxes if not len(prev) else prev

    boxes = boxes[:, 4:]

    return boxes[:num_boxes]


def scaled_crops(fg, alpha, num_crops=TOTAL_BOXES, repeats=7):
    maxangle = max_angle(fg)
    rotates = round(
        maxangle * (repeats - 1) / 45.0
    )  # TODO: check and make more rotates
    angles = np.random.uniform(
        -maxangle, maxangle, size=rotates
    ).tolist()  # TODO: linspace

    # TODO targ_actual = min(targ_actual, 2100) and no more then max test size?

    crops = []
    for i in range(repeats):
        if i < rotates:
            fg_, alpha_, full_ = random_rotate(fg, alpha, angles[i])
        else:
            fg_, alpha_, full_ = fg, alpha, np.ones_like(alpha, "bool")

        boxes = crop_boxes(alpha_, full_, num_crops)
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

    random.shuffle(crops)
    crops = crops[:num_crops]
    assert len(crops) == num_crops

    return crops


def crop_augment(fg, alpha, replay=False):
    interpolations = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    if max(fg.shape[:2]) > CROP_SIZE:
        interpolations += [cv2.INTER_AREA]
    interpolations = np.random.choice(interpolations, size=2)
    fg = cv2.resize(fg, (CROP_SIZE, CROP_SIZE), interpolation=interpolations[0])
    alpha = cv2.resize(
        alpha, (CROP_SIZE, CROP_SIZE), interpolation=interpolations[1]
    )

    compose_cls = alb.ReplayCompose if replay else alb.Compose
    aug = compose_cls(
        [
            # Color
            alb.RandomGamma(
                p=0.5
            ),  # reported as most useful # TODO: on-the-fly?
            alb.OneOf(
                [
                    alb.CLAHE(),
                    # alb.OneOf([
                    #   alb.ChannelDropout(fill_value=value)
                    #   for value in range(256)]), # disable for matting
                    # alb.ChannelShuffle(), # on-the-fly
                    # alb.ColorJitter(), # on-the-fly
                    # alb.OneOf([
                    #   alb.Equalize(by_channels=value)
                    #   for value in [True, False]]), # disable for matting
                    alb.FancyPCA(),
                    # alb.PixelDropout(), # disable for matting
                    alb.RGBShift(),
                    alb.RandomToneCurve(),
                    alb.Sharpen(alpha=(0.1, 0.4), p=0.1),
                    # alb.ToGray(p=0.1), # disable for matting
                    # alb.ToSepia(p=0.05), # disable for matting
                    alb.UnsharpMask(p=0.1),
                ],
                p=0.4,
            ),
            # Blur
            alb.OneOf(
                [
                    alb.Blur(blur_limit=(3, 4)),
                    alb.GaussianBlur(blur_limit=(3, 5)),
                    alb.MedianBlur(blur_limit=3),
                    alb.MotionBlur(blur_limit=(3, 10)),
                    alb.GlassBlur(max_delta=2, iterations=1, p=0.1),
                ],
                p=0.2,
            ),
            # Noise
            alb.OneOf(
                [
                    alb.GaussNoise(var_limit=(10.0, 500.0)),
                    alb.ISONoise(color_shift=(0.0, 0.1), intensity=(0.1, 0.7)),
                    alb.OneOf(
                        [
                            alb.MultiplicativeNoise(
                                multiplier=(0.95, 1.05), per_channel=value
                            )
                            for value in [True, False]
                        ],
                        p=0.2,
                    ),
                    # TODO: check
                    # alb.ImageCompression(
                    #   quality_lower=70, quality_upper=99), # on-the-fly
                    # alb.Posterize(num_bits=(6, 8)), # disable for matting
                ],
                p=0.1,
            ),
        ]
    )

    augmented = aug(image=fg, mask=alpha)
    assert (augmented["mask"] == alpha).all()

    if replay:
        print(drop_unapplied(augmented["replay"]))

    return augmented["image"], alpha


class MattingDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    def __init__(
        self,
        *,
        source_dirs,
        background_dirs,
        data_dir,
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
        source_dirs = [os.fspath(s) for s in source_dirs]

        bad = [s for s in source_dirs if not os.path.isdir(s)]
        if bad:
            raise ValueError(
                "Some of source directories do not exist: {}".format(bad)
            )

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
                        encoding_format="jpeg",
                    ),
                    "background": tfds.features.Image(
                        shape=(None, None, 3),
                        dtype=tf.uint8,
                        encoding_format="jpeg",
                    ),
                    "trimap": tfds.features.Image(
                        shape=(None, None, 1),
                        dtype=tf.uint8,
                        encoding_format="jpeg",
                    ),
                }
            ),
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
                raise ValueError("A list expected for background directories")

            bg_bad = [s for s in bg_dirs if not os.path.isdir(s)]
            if bg_bad:
                raise ValueError(
                    "Some of background directories do not exist: {}".format(
                        bg_bad
                    )
                )

            bg_dirs = [os.fspath(s) for s in bg_dirs]
            for d in bg_dirs:
                for root, _, files in os.walk(d):
                    for file in files:
                        if file[-4:] not in {".jpg", "jpeg", ".png"}:
                            continue
                        self.bg_files.append(os.path.join(root, file))

            if not self.bg_files:
                raise ValueError(f"No backgrounds found in {bg_dirs}")

            random.shuffle(self.bg_files)

        for alpha_file in self._iterate_source(training):
            for key, alpha, fg, bg, trimap in self._transform_example(
                alpha_file, training
            ):
                alpha = np.squeeze(alpha)[..., None]

                yield key, {
                    "alpha": alpha,
                    "foreground": fg,
                    "background": bg,
                    "trimap": trimap,
                }

    def _iterate_source(self, training):
        for source_dir in self.source_dirs:
            for dirpath, _, filenames in os.walk(source_dir):
                for file in filenames:
                    if not file.endswith("similar.json"):
                        continue

                    with open(os.path.join(dirpath, file), "rt") as f:
                        self.similar.update(json.load(f))

        for source_dir in self.source_dirs:
            for dirpath, _, filenames in os.walk(source_dir):
                for file in filenames:
                    if training == bool(
                        re.search(
                            self.test_re, os.path.join("/", dirpath, file)
                        )
                    ):
                        continue

                    alpha_ext = "-alpha.png"
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

        interpolations = [
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ]
        if min(bg.shape[:2]) >= CROP_SIZE:
            max_scale = min(bg.shape[:2]) / CROP_SIZE
            curr_scale = np.random.uniform(
                low=1.0, high=max_scale, size=1
            ).item()
            aug = alb.Compose(
                [
                    alb.RandomCrop(
                        round(CROP_SIZE * curr_scale),
                        round(CROP_SIZE * curr_scale),
                        p=1,
                    ),
                    alb.OneOf(
                        [
                            alb.SmallestMaxSize(CROP_SIZE, interpolation=value)
                            for value in interpolations
                        ],
                        p=1,
                    ),
                ]
            )
        else:
            aug = alb.Compose(
                [
                    alb.OneOf(
                        [
                            alb.SmallestMaxSize(CROP_SIZE, interpolation=value)
                            for value in interpolations
                        ],
                        p=1,
                    ),
                    alb.RandomCrop(CROP_SIZE, CROP_SIZE, p=1),
                ]
            )
        augmented = aug(image=bg)

        return augmented["image"]

    def _transform_example(self, alpha_file, training):
        fg_file = alpha_file.replace("-alpha.png", "-fg.jpg")
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

                    trimap_ = np.zeros((1, 1, 1), dtype="uint8")
                    assert fg_.shape[:2] == alpha_.shape[:2] == bg_.shape[:2], (
                        fg_.shape[:2],
                        alpha_.shape[:2],
                        bg_.shape[:2],
                    )

                    yield "{}_{}_{}".format(
                        alpha_file, i, j
                    ), alpha_, fg_, bg_, trimap_
        else:
            for i in range(100):
                bg_file = alpha_file.replace(
                    "-alpha.png", "-{}_bg.jpg".format(str(i).zfill(2))
                )
                trimap_file = alpha_file.replace(
                    "-alpha.png", "-{}_trimap.png".format(str(i).zfill(2))
                )
                if not os.path.isfile(bg_file) or not os.path.isfile(
                    trimap_file
                ):
                    continue

                bg = cv2.cvtColor(cv2.imread(bg_file), cv2.COLOR_BGR2RGB)
                trimap = cv2.imread(trimap_file, cv2.IMREAD_GRAYSCALE)[
                    ..., None
                ]
                assert alpha.shape[:2] == bg.shape[:2], alpha_file
                assert alpha.shape[:2] == trimap.shape[:2], alpha_file
                assert not len(set(np.unique(trimap)) - {0, 128, 255})

                yield "{}_{}".format(alpha_file, i), alpha, fg, bg, trimap


@tf.function(jit_compile=False)
def _augment_examples(examples):
    alpha = examples["alpha"]
    foreground = examples["foreground"]
    background = examples["background"]

    alpha.set_shape(alpha.shape[:1] + [CROP_SIZE, CROP_SIZE, 1])
    foreground.set_shape(foreground.shape[:1] + [CROP_SIZE, CROP_SIZE, 3])
    background.set_shape(background.shape[:1] + [CROP_SIZE, CROP_SIZE, 3])

    alpha = matting.augment_alpha(alpha)
    # foreground, alpha = matting.random_compose(
    #   foreground, alpha, trim=(max(TRIMAP_SIZE), 0.95), solve=False)
    foreground, [alpha], _ = rand_augment_matting(foreground, [alpha], None)
    # foreground = matting.solve_fg(
    #   foreground, alpha, kappa=0.334, steps=3)  # for crop size 512
    background = tf.random.shuffle(background)
    trimap = matting.alpha_trimap(alpha, size=TRIMAP_SIZE)
    trimap = matting.augment_trimap(trimap)

    return {
        "alpha": alpha,
        "foreground": foreground,
        "background": background,
        "trimap": trimap,
    }


@tf.function(jit_compile=True)
def _prepare_examples_mf(examples):
    alpha = ops.cast(examples["alpha"], "float32") / 255.0
    foreground = ops.cast(examples["foreground"], "float32") / 255.0
    background = ops.cast(examples["background"], "float32") / 255.0

    image = foreground * alpha + background * (1.0 - alpha)
    image = ops.cast(ops.round(image * 255.0), "uint8")

    features = {"image": image, "trimap": examples["trimap"]}
    labels = ops.concatenate([alpha, foreground, background], axis=-1)
    weights = ops.cast(examples["trimap"] == 128, "float32")

    return (
        features,
        (alpha, labels, labels, labels),
        (weights, None, None, None),
    )


@tf.function(jit_compile=False)
def _prepare_examples_fba(examples):
    alpha = ops.cast(examples["alpha"], "float32") / 255.0
    foreground = ops.cast(examples["foreground"], "float32") / 255.0
    background = ops.cast(examples["background"], "float32") / 255.0

    image = foreground * alpha + background * (1.0 - alpha)
    image = ops.cast(ops.round(image * 255.0), "uint8")

    twomap = twomap_transform(examples["trimap"])
    distance = distance_transform(examples["trimap"])

    features = {"image": image, "twomap": twomap, "distance": distance}

    alfgbg = ops.concatenate([alpha, foreground, background], axis=-1)
    labels = (alfgbg, alpha, foreground, background)

    weight = ops.cast(examples["trimap"] == 128, "float32")
    weights = (None, weight, None, None)

    return features, labels, weights


@tf.function(jit_compile=True)
def _normalize_trimap(examples):
    trimap = examples["trimap"]
    trimap = ops.cast(trimap // 86, "int32") * 128
    trimap = ops.cast(ops.clip(trimap, 0, 255), "uint8")

    examples = {
        "alpha": examples["alpha"],
        "foreground": examples["foreground"],
        "background": examples["background"],
        "trimap": trimap,
    }

    return examples


def make_dataset(data_dir, split_name, out_mode, batch_size=1):
    train_split = tfds.Split.TRAIN == split_name

    builder = MattingDataset(
        source_dirs=[], background_dirs=[], data_dir=data_dir
    )
    builder.download_and_prepare()

    dataset = builder.as_dataset(
        split=split_name, batch_size=None, shuffle_files=train_split
    )

    if train_split:
        dataset = (
            dataset.shuffle(batch_size * 8)
            .batch(max(32, batch_size * 4), drop_remainder=True)
            .map(
                _augment_examples,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .unbatch()
            .shuffle(batch_size * 8)
            .batch(batch_size, drop_remainder=True)
        )
    else:
        dataset = dataset.batch(1).map(
            _normalize_trimap, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    if "fba" == out_mode:
        dataset = dataset.map(
            _prepare_examples_fba,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    elif "mf" == out_mode:
        dataset = dataset.map(
            _prepare_examples_mf,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    else:
        raise ValueError("Unknown mode")

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
