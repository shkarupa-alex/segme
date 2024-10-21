import functools
import io
import os
import resource
from functools import partial

import cv2
import nltk
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.src import ops
from keras.src.applications import imagenet_utils
from keras.src.dtype_policies import dtype_policy
from keras.src.utils import file_utils

from segme.model.classification.tree import flat21841_class_map
from segme.model.classification.tree import flat21843_class_map
from segme.model.classification.tree import synsets_1k_21k
from segme.model.classification.tree import tree_class_map
from segme.utils.common import cut_mix_up
from segme.utils.common import rand_augment_full


class Imagenet21k1k(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    CORRUPTED_FILES = {
        "n01678043_6448.JPEG",
        "n01896844_997.JPEG",
        "n02368116_318.JPEG",
        "n02428089_710.JPEG",
        "n02487347_1956.JPEG",
        "n02597972_5463.JPEG",
        "n03957420_33553.JPEG",
        "n03957420_30695.JPEG",
        "n03957420_8296.JPEG",
        "n04135315_9318.JPEG",
        "n04135315_8814.JPEG",
        "n04257684_9033.JPEG",
        "n04427559_2974.JPEG",
        "n06470073_47249.JPEG",
        "n07930062_4147.JPEG",
        "n09224725_3995.JPEG",
        "n09359803_8155.JPEG",
        "n09894445_7463.JPEG",
        "n12353203_3849.JPEG",
        "n12630763_8018.JPEG",
    }

    def __init__(
        self,
        *,
        archive21k=None,
        archive1k=None,
        image_size=384,
        crop_pct=0.875,
        **kwargs,
    ):
        self.archive21k = archive21k
        self.archive1k = archive1k
        self.image_size = image_size
        self.crop_pct = crop_pct

        self._label_to_synset_cache = {}
        self._val_to_label_labels = []

        super().__init__(**kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(
                        shape=(None, None, 3), encoding_format="jpeg"
                    ),
                    "file": tfds.features.Text(),
                    "in1k": tfds.features.Scalar(dtype=tf.bool),
                    "size": tfds.features.Text(),
                    "label": tfds.features.Text(),
                    "synset": tfds.features.Text(),
                    "class": tfds.features.ClassLabel(names=synsets_1k_21k()),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return {
            tfds.Split.VALIDATION: self._generate_examples(False),
            tfds.Split.TRAIN: self._generate_examples(True),
        }

    def _generate_examples(self, training):
        _, high = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

        skip_files = set()

        for key, features in self._generate_examples_1k(training):
            if training:
                skip_files.add(features["file"])

            yield key, features

        if training and self.archive21k is not None:
            for key, features in self._generate_examples_21k(skip_files):
                yield key, features

    def _generate_examples_1k(self, training):
        if self.archive1k is None:
            raise ValueError(
                "ImageNet1k archive path must be set to generate dataset."
            )

        if not self.archive1k.endswith(".tar.gz"):
            raise ValueError(
                "ImageNet1k archive has wrong extension (expected .tar.gz)."
            )

        if not file_utils.exists(self.archive1k):
            raise ValueError(
                f"ImageNet1k archive {self.archive1k} is not a file."
            )

        for file_path, file_obj in tfds.download.iter_archive(
            self.archive1k, tfds.download.ExtractMethod.TAR_GZ_STREAM
        ):

            if not file_path.endswith(".JPEG"):
                continue
            if training and "/train/" not in file_path:
                continue
            if not training and "/val/" not in file_path:
                continue

            if training:
                label_name = file_path.split("/train/")[-1].split("/")[0]
            else:
                label_name = self._label_from_valname(file_path)
            synset_name = self._synset_from_label(label_name)

            file_name = file_path.split("/")[-1]
            if file_name in self.CORRUPTED_FILES:
                continue

            image, size = self._prepare_image(file_obj)
            if image is None:
                continue

            yield file_name, {
                "image": image,
                "file": file_name,
                "in1k": True,
                "size": size,
                "label": label_name,
                "synset": synset_name,
                "class": synset_name,
            }

    def _generate_examples_21k(self, skip_files):
        if self.archive21k is None:
            raise ValueError(
                "ImageNet21k archive path must be set to generate dataset."
            )

        if not self.archive21k.endswith(".tar.gz"):
            raise ValueError(
                "ImageNet21k archive has wrong extension (expected .tar.gz)."
            )

        if not file_utils.exists(self.archive21k):
            raise ValueError(
                f"ImageNet21k archive {self.archive21k} is not a file."
            )

        for arch_path, arch_obj in tfds.download.iter_archive(
            self.archive21k, tfds.download.ExtractMethod.TAR_GZ_STREAM
        ):

            if not arch_path.endswith(".tar"):
                continue

            label_name = arch_path.split("/")[-1].replace(".tar", "")
            synset_name = self._synset_from_label(label_name)

            for file_path, file_obj in tfds.download.iter_archive(
                io.BytesIO(arch_obj.read()),
                tfds.download.ExtractMethod.TAR_STREAM,
            ):

                if not file_path.endswith(".JPEG"):
                    continue

                file_name = file_path.split("/")[-1]
                if file_name in self.CORRUPTED_FILES:
                    continue
                if file_name in skip_files:
                    continue

                image, size = self._prepare_image(file_obj)
                if image is None:
                    continue

                yield file_name, {
                    "image": image,
                    "file": file_name,
                    "in1k": False,
                    "size": size,
                    "label": label_name,
                    "synset": synset_name,
                    "class": synset_name,
                }

    @functools.lru_cache(maxsize=20000)
    def _synset_from_label(self, label):
        if not self._label_to_synset_cache:
            nltk.download("omw-1.4")
            nltk.download("wordnet")

        if label not in self._label_to_synset_cache:
            pos, index = label[0], int(label[1:].lstrip("0"))
            assert "n" == pos, label

            synset = nltk.corpus.wordnet.synset_from_pos_and_offset(pos, index)
            self._label_to_synset_cache[label] = synset.name()

        return self._label_to_synset_cache[label]

    def _label_from_valname(self, file_path):
        if not self._val_to_label_labels:
            imagenet_common = tfds.datasets.imagenet2012.imagenet_common
            labels_path = imagenet_common._VALIDATION_LABELS_FNAME
            labels_path = tfds.core.tfds_path(labels_path)

            with tf.io.gfile.GFile(os.fspath(labels_path)) as f:
                self._val_to_label_labels = f.read().strip().splitlines()

        assert "_val_" in file_path, file_path
        index = file_path.split("_val_")[-1].replace(".JPEG", "")
        index = int(index.lstrip("0")) - 1

        return self._val_to_label_labels[index]

    def _prepare_image(self, file_obj):
        image = np.frombuffer(file_obj.read(), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image is None:
            return None, None

        size = f"{image.shape[0]}x{image.shape[1]}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, size


@tf.function(jit_compile=False)
def _resize_crop(example, size, train, crop_pct=0.875):
    image = example["image"]
    shape = ops.cast(ops.shape(image)[:2], "float32")

    if train:
        crop = ops.random.uniform(
            [2], minval=ops.min(shape) * crop_pct, maxval=shape
        )
        crop = ops.cast(ops.round(crop), "int32")
        crop = ops.concatenate([crop, np.array([3], "int32")], axis=-1)

        image = tf.image.random_crop(image, crop)
        image = ops.image.resize(
            image,
            [size, size],
            interpolation="bicubic",
            antialias=True,
        )
    else:
        shape_ = shape * size / crop_pct / ops.min(shape)
        shape_ = ops.cast(ops.round(shape_), "int32")
        shape_ = ops.unstack(shape_)

        image = ops.image.resize(
            image, shape_, interpolation="bicubic", antialias=True
        )

        crop_h = (shape_[0] - size) // 2
        crop_w = (shape_[1] - size) // 2
        image = image[crop_h : crop_h + size, crop_w : crop_w + size]

    image = ops.clip(image, 0, 255)
    image = ops.cast(ops.round(image), "uint8")
    image.set_shape([size, size, 3])  # TODO

    return image, example["class"]


@tf.function(jit_compile=False)
def _transform_examples(
    images,
    labels,
    augment,
    levels,
    magnitude,
    cutmixup,
    classes,
    preprocess,
    remap,
):
    if remap:
        labels = remap.lookup(labels)

    if augment:
        images = tf.image.convert_image_dtype(images, "float32")
        images, _, _ = rand_augment_full(images, None, None, levels, magnitude)
        if cutmixup:
            images, labels, _ = cut_mix_up(
                images,
                labels,
                None,
                classes,
                cutmix_prob=magnitude * 0.5,
                mixup_prob=magnitude * 0.5,
            )
        images = ops.clip(images, 0.0, 1.0)
        images = ops.round(images * 255.0)

    if preprocess:
        images = ops.cast(images, dtype_policy.dtype_policy().compute_dtype)
        images = imagenet_utils.preprocess_input(images, mode=preprocess)
    elif augment:
        images = ops.cast(images, "uint8")

    return images, labels


def make_dataset(
    data_dir,
    split_name,
    batch_size,
    num_repeats=1,
    image_size=384,
    preprocess_mode=None,
    aug_levels=5,
    aug_magnitude=0.5,
    use_cutmixup=False,
    remap_classes=None,
):
    train_split = tfds.Split.TRAIN == split_name

    builder = Imagenet21k1k(data_dir=data_dir)
    builder.download_and_prepare()

    dataset = builder.as_dataset(
        split=split_name, batch_size=None, shuffle_files=train_split
    )

    resize_crop = partial(_resize_crop, size=image_size, train=train_split)
    dataset = dataset.map(
        resize_crop,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).batch(batch_size, drop_remainder=train_split)

    if train_split:
        max_pixels_32_gb = 256**3 // 2
        dataset = dataset.shuffle(max_pixels_32_gb // (image_size**2))

    class_map, class_size = None, len(synsets_1k_21k())
    if remap_classes:
        if "tree" == remap_classes:
            class_map = tree_class_map()
        elif "flat1" == remap_classes:
            class_map = flat21841_class_map()
        elif "flat3" == remap_classes:
            class_map = flat21843_class_map()
        else:
            raise ValueError("Unknown class remapping mode")
        class_size = len(set(class_map.values()))
        map_keys, map_values = zip(*class_map.items())
        map_init = tf.lookup.KeyValueTensorInitializer(
            map_keys, map_values, "int64", "int64"
        )
        class_map = tf.lookup.StaticHashTable(map_init, -1)

    train_augment = train_split and aug_levels and aug_magnitude
    transform_examples = partial(
        _transform_examples,
        augment=train_augment,
        levels=aug_levels,
        magnitude=aug_magnitude,
        cutmixup=use_cutmixup,
        classes=class_size,
        preprocess=preprocess_mode,
        remap=class_map,
    )
    if train_augment or preprocess_mode or class_map:
        dataset = dataset.map(
            transform_examples,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    if num_repeats > 1:
        dataset = dataset.repeat(num_repeats)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
