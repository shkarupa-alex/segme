import albumentations as alb
import cv2
import numpy as np
import os
import re
import tensorflow as tf
import tensorflow_datasets as tfds
from operator import itemgetter
from segme.utils.albumentations import drop_unapplied
from segme.utils.common import augment_onthefly
from segme.utils.matting.np import alpha_trimap as alpha_trimap_np
from segme.utils.matting.tf import alpha_trimap as alpha_trimap_tf

MIN_SIZE = 384
BUCKET_GROUPS = {
    (384, 512): (384, 512), (384, 576): (384, 576), (576, 384): (576, 384), (384, 672): (384, 672),
    (384, 640): (384, 672), (384, 608): (384, 608), (512, 384): (512, 384), (384, 384): (384, 384),
    (384, 544): (384, 544), (384, 480): (384, 480), (480, 384): (480, 384), (544, 384): (544, 384),
    (384, 448): (384, 448), (384, 416): (384, 448), (448, 384): (448, 384), (416, 384): (448, 384),
    (608, 384): (608, 384), (672, 384): (672, 384), (640, 384): (672, 384), (704, 384): (672, 384),
    (384, 704): (384, 704), (384, 736): (384, 704), (384, 768): (384, 704), (736, 384): (736, 384),
    (768, 384): (736, 384)}

SORTED_GROUPS = sorted(set(BUCKET_GROUPS.values()), key=lambda v: v[0] * v[1])
GROUPS_LENGTH = {bg: i + 1 for i, bg in enumerate(SORTED_GROUPS)}


def target_size(size):
    size = 8 * (size[0] // 8), 8 * (size[1] // 8)
    bucket_keys = list(BUCKET_GROUPS.keys())
    if size not in bucket_keys:
        dist = [((known[0] - size[0]) ** 2 + (known[1] - size[1]) ** 2) ** 0.5 for known in bucket_keys]
        minid, _ = min(enumerate(dist), key=itemgetter(1))
        size = bucket_keys[minid]

    return BUCKET_GROUPS[size]


def target_length(size):
    group = BUCKET_GROUPS[size]
    length = GROUPS_LENGTH[group]

    return length


def bucket_batch(max_pixels):
    buck_bounds, batch_sizes = [], []
    for group in SORTED_GROUPS:
        length = GROUPS_LENGTH[group]
        buck_bounds.append(length)

        batch = max_pixels / (group[0] * group[1])
        batch = int(batch) if batch <= 8 else 8 * int(batch // 8)
        batch_sizes.append(batch)

    min_idx = len(batch_sizes) - batch_sizes[::-1].index(min(batch_sizes)) - 1
    skip_len = max(buck_bounds) + 1 if batch_sizes[min_idx] > 0 else buck_bounds[min_idx]
    batch_sizes = [max(b, 1) for b in batch_sizes[:1] + batch_sizes]

    return buck_bounds, batch_sizes, skip_len


def apply_scale(image, mask, depth, trimap, size):
    if isinstance(size, int):
        scale = size / min(mask.shape[:2])
        size = round(mask.shape[0] * scale), round(mask.shape[1] * scale)
    size = size[::-1]

    interpolation = cv2.INTER_AREA
    if min(image.shape[:2]) < min(size):
        interpolation = cv2.INTER_LANCZOS4

    _image = cv2.resize(image, size, interpolation=interpolation)
    _mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST_EXACT)
    _depth = cv2.resize(depth, size, interpolation=interpolation)
    _trimap = cv2.resize(trimap, size, interpolation=cv2.INTER_NEAREST_EXACT)

    return _image, _mask, _depth, _trimap


def trimap_size(mask, size, border):
    if isinstance(size, tuple) and 2 == len(size):
        size = (size[0] + size[1]) / 2.

    if not isinstance(size, (int, float)):
        raise ValueError('Expecting `size` to be a single size or a tuple of [height; width].')

    return round(border * (mask.shape[0] + mask.shape[1]) / (size * 2.))


def mask_trimap(mask, size):
    if len(mask.shape) not in {2, 3}:
        raise ValueError('Expecting `mask` rank to be 2 or 3.')

    if 3 == len(mask.shape) and 1 != mask.shape[-1]:
        raise ValueError('Expecting `mask` channels size to be 1.')

    if 'uint8' != mask.dtype:
        raise ValueError('Expecting `mask` dtype to be `uint8`.')

    if isinstance(size, tuple) and 2 == len(size):
        size = np.random.randint(size[0], size[1] + 1, 1, dtype='int32').item()

    if not isinstance(size, int):
        raise ValueError('Expecting `size` to be a single margin or a tuple of [min; max] margins.')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(mask, kernel, iterations=size)
    eroded = cv2.erode(mask, kernel, iterations=size)

    trimap = np.full(mask.shape, 128, dtype=mask.dtype)
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0

    return trimap


def train_augment(image, mask, depth, trimap, replay=False):
    src_size = mask.shape
    trg_size = target_size(src_size)

    compose_cls = alb.ReplayCompose if replay else alb.Compose
    aug = compose_cls([
        # Color
        alb.OneOf([
            alb.CLAHE(),
            alb.OneOf([
                alb.ChannelDropout(fill_value=value) for value in range(256)]),
            # alb.ChannelShuffle(), # on-the-fly
            alb.ColorJitter(),
            alb.OneOf([
                alb.Equalize(by_channels=value) for value in [True, False]]),
            alb.FancyPCA(),
            # alb.HueSaturationValue(), # on-the-fly
            alb.PixelDropout(),
            alb.RGBShift(),
            # alb.RandomBrightnessContrast(), # on-the-fly
            alb.RandomGamma(),
            alb.RandomToneCurve(),
            alb.Sharpen(),
            alb.ToGray(p=0.1),
            alb.ToSepia(p=0.05),
            alb.UnsharpMask()
        ], p=0.7),

        # Blur
        alb.OneOf([
            alb.AdvancedBlur(),
            alb.Blur(blur_limit=(3, 5)),
            alb.GaussianBlur(blur_limit=(3, 5)),
            alb.MedianBlur(blur_limit=5),
            alb.RingingOvershoot(blur_limit=(3, 5)),
        ], p=0.2),

        # Noise
        alb.OneOf([
            alb.Downscale(scale_max=0.75, interpolation=cv2.INTER_NEAREST_EXACT),
            alb.GaussNoise(var_limit=(10.0, 100.0)),
            alb.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 0.7)),
            alb.ImageCompression(quality_lower=25, quality_upper=95),
            alb.OneOf([
                alb.MultiplicativeNoise(per_channel=value1, elementwise=value2)
                for value1, value2 in [(True, True), (True, False), (False, True), (False, False)]]),
        ], p=0.3),

        # Distortion and scaling
        alb.OneOf([
            alb.Affine(),
            alb.ElasticTransform(alpha_affine=25, border_mode=cv2.BORDER_CONSTANT),
            alb.GridDistortion(border_mode=cv2.BORDER_CONSTANT),
            # makes image larger in all directions
            # alb.OpticalDistortion(distort_limit=(-0.5, 0.5), border_mode=cv2.BORDER_CONSTANT),
            # alb.Perspective(scale=(0.01, 0.05)), # moves image to top-left corner
            alb.PiecewiseAffine(scale=(0.01, 0.03)),
        ], p=0.2),

        # Rotate
        alb.Rotate(limit=(-45, 45), border_mode=cv2.BORDER_CONSTANT, p=0.2),

        # Pad & crop
        alb.PadIfNeeded(*trg_size, border_mode=cv2.BORDER_CONSTANT, p=1),
        alb.RandomCrop(*trg_size, p=1),
    ], additional_targets={'depth': 'mask', 'trimap': 'mask', 'weight': 'mask'})

    augmented = aug(image=image, mask=mask, depth=depth, trimap=trimap, weight=np.ones_like(mask))

    if augmented['mask'].shape != trg_size:
        raise ValueError('Wrong size after augmntation')

    if replay:
        print(drop_unapplied(augmented['replay']))

    return augmented['image'], augmented['mask'], augmented['depth'], augmented['trimap'], augmented['weight']


def test_augment(image, mask, depth, trimap):
    src_size = mask.shape
    trg_size = target_size(src_size)

    aug = alb.Compose([
        # Crop
        alb.PadIfNeeded(*trg_size, border_mode=cv2.BORDER_CONSTANT, p=1),
        alb.RandomCrop(*trg_size, p=1),
    ], additional_targets={'depth': 'mask', 'trimap': 'mask', 'weight': 'mask'})

    augmented = aug(image=image, mask=mask, depth=depth, trimap=trimap, weight=np.ones_like(mask))

    if augmented['mask'].shape != trg_size:
        raise ValueError('Wrong size after augmntation')

    return augmented['image'], augmented['mask'], augmented['depth'], augmented['trimap'], augmented['weight']


class SaliencyDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {'1.0.0': 'Initial release.'}

    def __init__(
            self, *, source_dirs, data_dir, train_aug=2, test_re='-test-', fix_size=None, set_weight=None,
            with_depth=False, with_trimap=False, config=None, version=None):
        if isinstance(source_dirs, str):
            source_dirs = [source_dirs]
        if not isinstance(source_dirs, list):
            raise ValueError('A list expected for source directories')

        source_dirs = [os.fspath(s) for s in source_dirs]

        bad_source = [s for s in source_dirs if not os.path.isdir(s)]
        if bad_source:
            raise ValueError('Some of source directories do not exist: {}'.format(bad_source))

        set_weight = {} if set_weight is None else dict(set_weight)
        if set_weight and (min(set_weight.values()) < 0 or max(set_weight.values()) > 255):
            raise ValueError('Weights should be in range [1; 255]')

        self.source_dirs = source_dirs
        self.train_aug = train_aug
        self.test_re = test_re
        self.fix_size = fix_size
        self.set_weight = set_weight
        self.with_depth = with_depth
        self.with_trimap = with_trimap

        super().__init__(data_dir=data_dir, config=config, version=version)

    def _info(self):
        size = (None, None) if isinstance(self.fix_size, int) else self.fix_size
        feats = {
            'image': tfds.features.Image(shape=size + (3,), dtype=tf.uint8, encoding_format='jpeg'),
            'mask': tfds.features.Image(shape=size + (1,), dtype=tf.uint8, encoding_format='jpeg'),
            'weight': tfds.features.Image(shape=size + (1,), dtype=tf.uint8, encoding_format='jpeg')
        }
        if isinstance(self.fix_size, int):
            feats['length'] = tfds.features.Tensor(shape=(), dtype=tf.int32)
        if self.with_depth:
            feats['depth'] = tfds.features.Image(shape=size + (1,), dtype=tf.uint8, encoding_format='jpeg')
        if self.with_trimap:
            feats['trimap'] = tfds.features.Image(shape=size + (1,), dtype=tf.uint8, encoding_format='jpeg')

        return tfds.core.DatasetInfo(
            builder=self,
            description='Saliency dataset',
            features=tfds.features.FeaturesDict(feats)
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            tfds.Split.TRAIN: self._generate_examples(True),
            tfds.Split.VALIDATION: self._generate_examples(False)
        }

    def _generate_examples(self, training):
        for image_file, mask_file, alpha_file, depth_file in self._iterate_source(training):
            for key, image, mask, depth, trimap, weight in self._transform_example(
                    image_file, mask_file, alpha_file, depth_file, training):

                sample_weight = 255.
                for weight_key, weight_value in self.set_weight.items():
                    if weight_key in mask_file:
                        sample_weight = 255. * float(weight_value) / max(self.set_weight.values())

                weight *= int(round(sample_weight))

                example = {
                    'image': image,
                    'mask': (mask >= 128).astype('uint8')[..., None],
                    'weight': weight[..., None]
                }
                if isinstance(self.fix_size, int):
                    example['length'] = target_length(mask.shape)
                if self.with_depth:
                    example['depth'] = depth[..., None]
                if self.with_trimap:
                    example['trimap'] = trimap[..., None]

                yield key, example

    def _iterate_source(self, training):
        for source_dir in self.source_dirs:
            for dirpath, _, filenames in os.walk(source_dir):
                for file in filenames:
                    image_ext = '-image.jpg'
                    if not file.endswith(image_ext):
                        continue
                    if '_' == file.split('/')[-1][0]:
                        continue

                    image_path = os.path.join(dirpath, file)
                    if training == bool(re.search(self.test_re, image_path)):
                        continue

                    mask_path = image_path.replace(image_ext, '-mask_manfix.')
                    if not os.path.exists(mask_path):
                        mask_path = image_path.replace(image_ext, '-mask_autofix.')
                    if not os.path.exists(mask_path):
                        mask_path = image_path.replace(image_ext, '-mask.png')

                    alpha_path = image_path.replace(image_ext, '-alpha.png')
                    if not os.path.exists(alpha_path):
                        alpha_path = image_path.replace(image_ext, '-alpha_auto.png')
                    if not os.path.exists(alpha_path):
                        alpha_path = None

                    depth_path = None
                    if self.with_depth:
                        depth_path = image_path.replace(image_ext, '-depth_dpt_m.png')
                        if not os.path.exists(depth_path):
                            depth_path = image_path.replace(image_ext, '-depth_dpt_l.png')
                        if not os.path.exists(depth_path):
                            depth_path = image_path.replace(image_ext, '-depth_dpt_h.png')
                        if not os.path.exists(depth_path):
                            depth_path = image_path.replace(image_ext, '-depth.png')

                    if os.path.exists(image_path.replace(image_ext, '-image_super.jpg')):
                        image_path = image_path.replace(image_ext, '-image_super.jpg')

                    yield image_path, mask_path, alpha_path, depth_path

    def _transform_example(self, image_file, mask_file, alpha_file, depth_file, training):
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        image_ = None
        if self.train_aug > 1 and '-image_super.' in image_file:
            image_ = cv2.cvtColor(cv2.imread(image_file.replace('-image_super.', '-image.')), cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        alpha = None if alpha_file is None else cv2.imread(alpha_file, cv2.IMREAD_GRAYSCALE)

        depth = np.zeros_like(mask, dtype='uint8')
        if self.with_depth:
            depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
            if depth is None or np.uint8 != depth.dtype:
                raise ValueError(f'Wrong depth values: {depth_file}')

        trimap = np.zeros_like(mask, dtype='uint8')
        if self.with_trimap:
            size = trimap_size(mask, 384, 5)
            trimap = mask_trimap(mask, size)
            assert 0 == len(set(trimap.ravel()) - {0, 128, 255})  # TODO
            if alpha is not None:
                atrimap = alpha_trimap_np(alpha, size)
                trimap = np.where(128 == atrimap, 128, trimap)
            assert 0 == len(set(trimap.ravel()) - {0, 128, 255})  # TODO

        if len(set(mask.reshape(-1)) - {0, 255}):
            raise ValueError(f'Wrong mask values: {mask_file}')

        if training:
            train_aug = self.train_aug

            if image_ is not None and train_aug > 1:
                train_aug -= 1
                image0, mask0, depth0, trimap0 = apply_scale(image_, mask, depth, trimap, self.fix_size)
                image1, mask1, depth1, trimap1, weight1 = train_augment(image0, mask0, depth0, trimap0)

                yield f'{mask_file}_source', image1, mask1, depth1, trimap1, weight1

            image0, mask0, depth0, trimap0 = apply_scale(image, mask, depth, trimap, self.fix_size)
            for i in range(train_aug):
                image1, mask1, depth1, trimap1, weight1 = train_augment(image0, mask0, depth0, trimap0)

                yield f'{mask_file}_{i}', image1, mask1, depth1, trimap1, weight1
        else:
            image0, mask0, depth0, trimap0 = apply_scale(image, mask, depth, trimap, self.fix_size)
            image1, mask1, depth1, trimap1, weight1 = test_augment(image0, mask0, depth0, trimap0)

            yield f'{mask_file}', image1, mask1, depth1, trimap1, weight1


@tf.function()  # jit_compile=True
def _transform_examples(examples, augment, with_depth, with_trimap, max_weight):
    images = examples['image']
    labels = tf.cast(examples['mask'] > 127, 'int32')
    weights = tf.cast(examples['weight'], 'float32') * (max_weight / 255.)

    masks = [labels, weights]
    if with_depth:
        masks.append(examples['depth'])

    if with_trimap:
        trimaps = examples['trimap']
        trimaps = tf.cast(trimaps // 86, 'int32') * 128
        trimaps = tf.cast(tf.clip_by_value(trimaps, 0, 255), 'uint8')
        trimaps = alpha_trimap_tf(trimaps, (0, 10))
        trimaps = tf.cast(trimaps, 'int32') // 86  # TODO: check
        masks.append(trimaps)

    if augment:
        images = tf.image.convert_image_dtype(images, 'float32')
        images, masks = augment_onthefly(images, masks)
        # TODO: https://github.com/tensorflow/tensorflow/pull/54484
        images = tf.cast(tf.round(tf.clip_by_value(images, 0., 1.) * 255.), 'uint8')

    features = {'image': images}
    targets, sample_weights, masks = masks[0:1], masks[1:2], masks[2:]

    if with_depth:
        depths, masks = masks[0], masks[1:]
        targets.append(depths)
        sample_weights.append(tf.cast(sample_weights[0] > 0., 'float32'))

    if with_trimap:
        targets.append(masks[0])
        sample_weights.append(sample_weights[0])

    if not with_depth and not with_trimap:
        return features, targets[0], sample_weights[0]

    return features, targets, sample_weights


def make_dataset(data_dir, split_name, with_depth, with_trimap, batch_pixels, fix_size=MIN_SIZE, max_weight=5.):
    builder = SaliencyDataset(
        source_dirs=[], data_dir=data_dir, fix_size=fix_size, with_depth=with_depth, with_trimap=with_trimap)
    builder.download_and_prepare()

    dataset = builder.as_dataset(split=split_name, batch_size=None, shuffle_files=True)

    if isinstance(fix_size, int):
        if fix_size != MIN_SIZE:
            raise ValueError(f'Fixed size {fix_size} not supported. Bucket groups pre-estimated for size: {MIN_SIZE}.')
        buck_bounds, batch_sizes, skip_len = bucket_batch(batch_pixels)
        dataset = dataset.filter(lambda example: example['length'] < skip_len)
        dataset = dataset.shuffle(max(batch_sizes) * 100)
        dataset = dataset.bucket_by_sequence_length(
            lambda example: example['length'],
            buck_bounds, batch_sizes, no_padding=True, drop_remainder=True)
    else:
        batch_size = batch_pixels // sum(fix_size)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.shuffle(batch_size * 100)

    dataset = dataset.map(
        lambda ex: _transform_examples(
            ex, tfds.Split.TRAIN == split_name, with_depth, with_trimap, max_weight),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
