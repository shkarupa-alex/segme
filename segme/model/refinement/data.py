import math
import os
import random
import re

import albumentations as alb
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.src import ops

from segme.common.impfunc import make_coords
from segme.utils.albumentations import drop_unapplied
from segme.utils.common import rand_augment_safe

CROP_SIZE = 512
IOU_MIN = 0.8
INTERPOLATIONS = [
    cv2.INTER_LINEAR,
    cv2.INTER_CUBIC,
    cv2.INTER_AREA,
    cv2.INTER_LANCZOS4,
]


def random_structure(size):
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size // 2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size // 2, size))


def random_dilate(seg, min_size=3, max_size=10):
    size = np.random.randint(min_size, max_size)
    kernel = random_structure(size)
    seg = cv2.dilate(seg, kernel, iterations=1)

    return seg


def random_erode(seg, min_size=3, max_size=10):
    size = np.random.randint(min_size, max_size)
    kernel = random_structure(size)
    seg = cv2.erode(seg, kernel, iterations=1)

    return seg


def compute_iou(seg, gt):
    intersection = seg * gt
    union = seg + gt

    return (np.count_nonzero(intersection) + 1e-6) / (
        np.count_nonzero(union) + 1e-6
    )


def perturb_segmentation(gt, iou_target):
    h, w = gt.shape[:2]
    seg = gt.copy()

    _, seg = cv2.threshold(seg, 127, 255, 0)

    # Rare case
    if h <= 2 or w <= 2:
        print("GT too small, returning original")
        return seg

    # Do a bunch of random operations
    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx + 1, w + 1), np.random.randint(
                ly + 1, h + 1
            )

            # Randomly set one pixel to 1/0. With the following dilate/erode,
            # we can create holes/external regions
            if np.random.rand() < 0.25:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])

        if compute_iou(seg, gt) < iou_target:
            break

    return seg


# def smooth_mask(mask):
#     height, width = mask.shape[:2]
#     scale = np.random.uniform(0.5, 1.0, None)
#     inter = np.random.choice([
#       cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])
#     mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale, interpolation=inter)


#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     repeat = np.random.randint(1, 4, 2)
#     dilated = cv2.dilate(mask, kernel, iterations=repeat[0])
#     eroded = cv2.erode(mask, kernel, iterations=repeat[1])

#     blurs = np.random.randint(2, 3, 6)
#     dilated = cv2.blur(dilated, tuple(blurs[0:2]))
#     eroded = cv2.blur(eroded, tuple(blurs[2:4]))


#     distt = np.random.choice([cv2.DIST_L1, cv2.DIST_L2, cv2.DIST_C])
#     distance = cv2.distanceTransform(dilated, distt, 3)
#     distance = np.where(eroded != 255, distance, 0.)
#     distance = cv2.normalize(distance, None, 0., 255., cv2.NORM_MINMAX)
#     distance = np.round(distance).astype('uint8')

#     combined = np.where(eroded != 255, distance, eroded)
#     combined = cv2.blur(combined, tuple(blurs[4:6]))
#     combined = cv2.resize(
#       combined, (width, height), interpolation=cv2.INTER_LINEAR)

#     return combined


def modify_boundary(
    image, regional_sample_rate=0.1, sample_rate=0.1, move_rate=0.0
):
    # Modifies boundary of the given mask:
    # - remove consecutive vertice of the boundary by regional sample rate
    # - remove any vertice by sample rate
    # - move vertice by distance between vertice and center of the mask by
    #   move rate

    iou_max = 1.0
    iou_target = np.random.rand() * (iou_max - IOU_MIN) + IOU_MIN

    # Get boundaries
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Only modified contours is needed actually
    sampled_contours = []
    modified_contours = []

    for contour in contours:
        if contour.shape[0] < 10:
            continue
        moments = cv2.moments(contour)

        # Remove region of contour
        number_of_vertices = contour.shape[0]
        number_of_removes = int(number_of_vertices * regional_sample_rate)

        idx_dist = []
        for i in range(number_of_vertices - number_of_removes):
            idx_dist.append(
                [i, np.sum((contour[i] - contour[i + number_of_removes]) ** 2)]
            )

        idx_dist = sorted(idx_dist, key=lambda x: x[1])

        remove_start = random.choice(
            idx_dist[: math.ceil(0.1 * len(idx_dist))]
        )[0]

        new_contour = np.concatenate(
            [
                contour[:remove_start],
                contour[remove_start + number_of_removes :],
            ],
            axis=0,
        )
        contour = new_contour

        # Sample contours
        number_of_vertices = contour.shape[0]
        indices = random.sample(
            range(number_of_vertices), int(number_of_vertices * sample_rate)
        )
        indices.sort()
        sampled_contour = contour[indices]
        sampled_contours.append(sampled_contour)

        modified_contour = np.copy(sampled_contour)
        if 0 != moments["m00"]:
            center = round(moments["m10"] / moments["m00"]), round(
                moments["m01"] / moments["m00"]
            )

            # Modify contours
            for idx, coor in enumerate(modified_contour):
                # 0.1 means change position of vertex to 10 percent farther
                # from center
                change = np.random.normal(0, move_rate)
                x, y = coor[0]
                new_x = x + (x - center[0]) * change
                new_y = y + (y - center[1]) * change

                modified_contour[idx] = [new_x, new_y]
        modified_contours.append(modified_contour)

    # Draw boundary
    gt = np.copy(image)
    image = np.zeros_like(image)

    modified_contours = [cont for cont in modified_contours if len(cont) > 0]
    if len(modified_contours) == 0:
        image = gt.copy()
    else:
        image = cv2.drawContours(image, modified_contours, -1, (255, 0, 0), -1)

    image = perturb_segmentation(image, iou_target)

    # if np.random.uniform(0., 1., None) > .5:  # extension by Shkarupa Alex
    #     image = smooth_mask(image)

    image = (image > 127).astype("uint8") * 255

    return image


def train_augment(image, mask, with_crops, replay=False):
    start_crop = CROP_SIZE * 2
    start_crop = min(start_crop, min(image.shape[:2]))
    trg_size = (CROP_SIZE, CROP_SIZE)

    aug = [
        # Crop or resize
        (
            alb.CropNonEmptyMaskIfExists(start_crop, start_crop, p=1)
            if with_crops
            else alb.OneOf(
                [
                    alb.LongestMaxSize(CROP_SIZE, interpolation=interpolation)
                    for interpolation in INTERPOLATIONS
                ],
                p=1,
            )
        ),
        # Color
        alb.OneOf(
            [
                alb.CLAHE(),
                alb.OneOf(
                    [
                        alb.ChannelDropout(fill_value=value)
                        for value in range(256)
                    ]
                ),
                # alb.ChannelShuffle(), # on-the-fly
                alb.ColorJitter(),
                alb.OneOf(
                    [alb.Equalize(by_channels=value) for value in [True, False]]
                ),
                alb.FancyPCA(),
                alb.HueSaturationValue(),
                alb.PixelDropout(),
                alb.RGBShift(),
                alb.RandomBrightnessContrast(),
                alb.RandomGamma(),
                alb.RandomToneCurve(),
                alb.Sharpen(),
                alb.ToGray(p=0.1),
                alb.ToSepia(p=0.05),
                alb.UnsharpMask(),
            ],
            p=0.7,
        ),
        # Blur
        alb.OneOf(
            [
                alb.AdvancedBlur(),
                alb.Blur(blur_limit=(3, 5)),
                alb.GaussianBlur(blur_limit=(3, 5)),
                alb.MedianBlur(blur_limit=5),
                alb.RingingOvershoot(blur_limit=(3, 5)),
            ],
            p=0.2,
        ),
        # Noise
        alb.OneOf(
            [
                alb.OneOf(
                    [
                        alb.Downscale(
                            scale_max=0.75, interpolation=interpolation
                        )
                        for interpolation in INTERPOLATIONS
                    ]
                ),
                alb.GaussNoise(var_limit=(10.0, 100.0)),
                alb.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 0.7)),
                alb.ImageCompression(quality_lower=25, quality_upper=95),
                alb.OneOf(
                    [
                        alb.MultiplicativeNoise(
                            per_channel=value1, elementwise=value2
                        )
                        for value1, value2 in [
                            (True, True),
                            (True, False),
                            (False, True),
                            (False, False),
                        ]
                    ]
                ),
            ],
            p=0.3,
        ),
        # Distortion and scaling
        alb.OneOf(
            [
                alb.OneOf(
                    [
                        alb.Affine(
                            mode=cv2.BORDER_REFLECT, interpolation=interpolation
                        )
                        for interpolation in INTERPOLATIONS
                    ]
                ),
                alb.OneOf(
                    [
                        alb.ElasticTransform(
                            border_mode=cv2.BORDER_REFLECT,
                            interpolation=interpolation,
                        )
                        for interpolation in INTERPOLATIONS
                    ]
                ),
                alb.OneOf(
                    [
                        alb.GridDistortion(
                            border_mode=cv2.BORDER_REFLECT,
                            interpolation=interpolation,
                        )
                        for interpolation in INTERPOLATIONS
                    ]
                ),
                # makes image larger in all directions
                # alb.OpticalDistortion(
                #   distort_limit=(-0.5, 0.5),
                #   border_mode=cv2.BORDER_REFLECT),
                # moves image to top-left corner
                # alb.Perspective(scale=(0.01, 0.05)),
                alb.OneOf(
                    [
                        alb.PiecewiseAffine(
                            scale=(0.01, 0.03),
                            mode="symmetric",
                            interpolation=interpolation,
                        )
                        for interpolation in INTERPOLATIONS
                    ]
                ),
            ],
            p=0.2,
        ),
        # Rotate
        alb.OneOf(
            [
                alb.Rotate(
                    limit=(-45, 45),
                    border_mode=cv2.BORDER_REFLECT,
                    interpolation=interpolation,
                )
                for interpolation in INTERPOLATIONS
            ],
            p=0.2,
        ),
        # Pad & crop
        alb.PadIfNeeded(*trg_size, border_mode=cv2.BORDER_REFLECT, p=1),
        alb.CropNonEmptyMaskIfExists(*trg_size, p=1),
    ]
    aug = (alb.ReplayCompose if replay else alb.Compose)(
        aug,
        additional_targets={"weight": "mask"},
    )

    augmented = aug(image=image, mask=mask, weight=np.ones_like(mask))

    if augmented["mask"].shape != trg_size:
        raise ValueError("Wrong size after augmntation")

    if replay:
        print(drop_unapplied(augmented["replay"]))

    return (
        augmented["image"],
        modify_boundary(augmented["mask"]),
        augmented["mask"],
        augmented["weight"],
    )


def valid_augment(image, mask, with_crops):
    trg_size = (CROP_SIZE, CROP_SIZE)
    if with_crops:
        aug = [
            (
                alb.OneOf(
                    [
                        alb.SmallestMaxSize(
                            CROP_SIZE, interpolation=interpolation, p=1
                        )
                        for interpolation in INTERPOLATIONS
                    ],
                    p=1,
                )
                if min(image.shape[:2]) < CROP_SIZE
                else alb.NoOp()
            ),
            alb.PadIfNeeded(*trg_size, border_mode=cv2.BORDER_REFLECT, p=1),
            alb.CropNonEmptyMaskIfExists(*trg_size, p=1),
        ]
    else:
        aug = [
            alb.OneOf(
                [
                    alb.LongestMaxSize(
                        CROP_SIZE, interpolation=interpolation, p=1
                    )
                    for interpolation in INTERPOLATIONS
                ],
                p=1,
            ),
            alb.PadIfNeeded(*trg_size, border_mode=cv2.BORDER_REFLECT, p=1),
        ]
    aug = alb.Compose(aug, additional_targets={"weight": "mask"})

    augmented = aug(image=image, mask=mask, weight=np.ones_like(mask))
    assert augmented["mask"].shape[:2] == trg_size

    return (
        augmented["image"],
        modify_boundary(augmented["mask"]),
        augmented["mask"],
        augmented["weight"],
    )


class RefineDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    def __init__(
        self,
        *,
        source_dirs,
        data_dir,
        with_crops=True,
        train_aug=1,
        test_re="-test-",
        set_weight=None,
        config=None,
        version=None,
    ):
        super().__init__(data_dir=data_dir, config=config, version=version)

        if isinstance(source_dirs, str):
            source_dirs = [source_dirs]
        if not isinstance(source_dirs, list):
            raise ValueError(
                "Expecting source directories to be a single path or a list "
                "of paths."
            )

        source_dirs = [os.fspath(s) for s in source_dirs]
        source_exists = [os.path.isdir(s) for s in source_dirs]
        if not all([True] + source_exists):
            raise ValueError("Some of source directories do not exist.")

        set_weight = {} if set_weight is None else dict(set_weight)
        if set_weight and (
            min(set_weight.values()) < 0.0 or max(set_weight.values()) > 5.0
        ):
            raise ValueError("Source weights should be in range [0.0; 5.0]")

        self.source_dirs = source_dirs
        self.with_crops = with_crops
        self.train_aug = train_aug
        self.test_re = test_re
        self.set_weight = set_weight
        self.empty_samples = 0
        self.full_samples = 0

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="Refine dataset",
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(
                        shape=(CROP_SIZE, CROP_SIZE, 3), encoding_format="jpeg"
                    ),
                    "mask": tfds.features.Image(
                        shape=(CROP_SIZE, CROP_SIZE, 1), encoding_format="jpeg"
                    ),
                    "label": tfds.features.Image(
                        shape=(CROP_SIZE, CROP_SIZE, 1), encoding_format="jpeg"
                    ),
                    "weight": tfds.features.Image(
                        shape=(CROP_SIZE, CROP_SIZE, 1), encoding_format="jpeg"
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
        for image_file, mask_file in self._iterate_source(training):
            for key, image, coarse, mask, weight in self._transform_example(
                image_file, mask_file, training
            ):
                yield key, {
                    "image": image,
                    "mask": coarse[..., None],
                    "label": mask[..., None],
                    "weight": weight[..., None],
                }

    def _iterate_source(self, training):
        if not self.source_dirs:
            raise ValueError(
                "Expecting source directories to contain at least one path "
                "for dataset generation."
            )

        for source_dir in self.source_dirs:
            if source_dir.split("/")[-1][0] in "._":
                continue

            for dirpath, _, filenames in os.walk(source_dir):
                # Skip some datasets
                if dirpath.split("/")[-1][0] in "._":
                    continue

                for file in filenames:
                    # Do not skip samples
                    # if file.split("/")[-1][0] in "._":
                    #     continue

                    image_ext = "-image.jpg"
                    if not file.endswith(image_ext):
                        continue

                    image_path = os.path.join(dirpath, file)

                    if training == bool(re.search(self.test_re, image_path)):
                        continue

                    # Do not use super resolution image to make task harder
                    # if (file.replace(image_ext, "-image_super.jpg")
                    #         in filenames):
                    #     image_path = image_path.replace(
                    #         image_ext, "-image_super.jpg"
                    #     )

                    mask = file.replace(image_ext, "-mask.png")
                    # Do not use manually fiexed masks
                    # if mask.replace("-mask.", "-mask_manfix.") in filenames:
                    #     mask = mask.replace("-mask.", "-mask_manfix.")

                    yield image_path, os.path.join(dirpath, mask)

    def _transform_example(self, image_file, mask_file, training):
        sample_weight = 1.0
        for weight_key, weight_value in self.set_weight.items():
            if weight_key in mask_file:
                sample_weight = weight_value

        if sample_weight < 1 / 20:
            return []

        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        if len(np.unique(mask)) == 1:
            return []

        if len(set(mask.reshape(-1)) - {0, 255}) != 0:
            raise ValueError(f"Wrong mask values in {mask_file}")

        do_aug = train_augment if training else valid_augment
        image_, mask_ = image, mask

        min_size = min(image_.shape[:2])
        if min_size < CROP_SIZE:
            if sample_weight * (min_size / CROP_SIZE) ** 2 < 1 / 20:
                return []

        max_scales = math.ceil(math.log2(min_size / CROP_SIZE))
        max_scales = min(5, max(1, max_scales))
        max_scales = max_scales if self.with_crops else 1
        assert min_size >= CROP_SIZE or 1 == max_scales, image_file

        for s in range(max_scales):
            curr_scale = 0.5**s
            curr_weight = 1.2**s

            interp = np.random.choice(INTERPOLATIONS)

            if min_size < CROP_SIZE:
                curr_scale = CROP_SIZE / min_size
                curr_weight *= (min_size / CROP_SIZE) ** 2

            image_ = cv2.resize(
                image,
                (0, 0),
                fx=curr_scale,
                fy=curr_scale,
                interpolation=interp,
            )
            assert min(image_.shape[:2]) >= CROP_SIZE, image_file

            mask_ = cv2.resize(
                mask,
                (image_.shape[1], image_.shape[0]),
                interpolation=cv2.INTER_NEAREST_EXACT,
            )
            assert image_.shape[:2] == mask_.shape[:2], image_file

            repeats = self.train_aug if training else 1
            repeats_ext = image_.shape[0] * image_.shape[1] / (CROP_SIZE**2)
            repeats_ext = max(1, int(repeats_ext**0.5))
            repeats_ext = repeats_ext if self.with_crops else 1

            for i in range(repeats * repeats_ext):
                image0, coarse0, mask0, weight0 = do_aug(
                    image_, mask_, self.with_crops
                )

                if (weight0 < 0.01).mean() > 0.33:
                    # Skip samples where > 1/3 of pixels are masked
                    continue

                # size0 = (mask0 == 255).mean()
                # if size0 < 0.05 or size0 > 0.95:
                #     continue
                #
                # iou0 = compute_iou(mask0, coarse0)
                # if iou0 < IOU_MIN:
                #     continue

                weight0 = weight0.astype("float32") * min(
                    sample_weight * curr_weight * 20.0, 255.0
                )
                weight0 = np.round(weight0).astype("uint8")
                if 0 == weight0.max():
                    continue

                assert min(image0.shape[:2]) == CROP_SIZE, image_file
                assert max(image0.shape[:2]) == CROP_SIZE, image_file

                yield "{}_{}_{}".format(
                    mask_file, s, i
                ), image0, coarse0, mask0, weight0


@tf.function(jit_compile=False)
def _transform_examples(
    examples, augment, batch_size, with_coord, with_prev, with_time
):
    images, masks, labels, weights = (
        examples["image"],
        examples["mask"],
        examples["label"],
        examples["weight"],
    )
    masks = ops.cast(masks > 127, "uint8") * 255

    if augment:
        images, [masks, labels], weights = rand_augment_safe(
            images, [masks, labels], weights, levels=3
        )

    features = {"image": images, "mask": masks}
    labels = ops.cast(labels > 127, "int32")
    weights = ops.cast(weights, "float32") / 20.0

    if with_coord:
        features["coord"] = make_coords(batch_size, CROP_SIZE, CROP_SIZE)

    if with_prev:
        features["prev"] = masks

    if with_time:
        time = ops.random.uniform([batch_size], maxval=with_time, dtype="int32")

        probs = tf.linspace(0.8, 0.0, 6)
        probs = ops.take(probs, time, axis=0)[:, None, None, None]
        noise = ops.random.uniform([batch_size] + masks.shape.as_list()[1:])
        trans = ops.cast(noise < probs, "uint8")
        sample = trans * ops.cast(labels, "uint8") * 255 + (1 - trans) * masks

        features["mask"] = sample
        features["time"] = time

    return features, labels, weights


def make_dataset(
    data_dir,
    split_name,
    batch_size,
    with_coord=False,
    with_prev=False,
    with_time=0,
):
    builder = RefineDataset(source_dirs=[], data_dir=data_dir)
    builder.download_and_prepare()

    dataset = builder.as_dataset(
        split=split_name, batch_size=None, shuffle_files=True
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if tfds.Split.TRAIN == split_name:
        dataset = dataset.shuffle(32)

    dataset = dataset.map(
        lambda ex: _transform_examples(
            ex,
            tfds.Split.TRAIN == split_name,
            batch_size,
            with_coord,
            with_prev,
            with_time,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
