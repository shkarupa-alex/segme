import os
import re
import resource

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.src.applications import imagenet_utils
from keras.src.mixed_precision import global_policy


class Clip(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    def __init__(
        self,
        *,
        source_dirs,
        data_dir,
        image_size=384,
        logits_size=1152,
        test_re="-val"
    ):
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
        self.image_size = image_size
        self.logits_size = logits_size
        self.test_re = test_re

        super().__init__(data_dir=data_dir)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(
                        shape=(self.image_size, self.image_size, 3),
                        encoding_format="jpeg",
                    ),
                    "logit": tfds.features.Tensor(
                        shape=(2, self.logits_size), dtype=tf.float32
                    ),
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

        for image_file in self._iterate_source(training):
            for key, image, logit in self._transform_example(image_file):
                yield key, {"image": image, "logit": logit}

    def _iterate_source(self, training):
        for source_dir in self.source_dirs:
            for dirpath, _, filenames in os.walk(source_dir):
                for file in filenames:
                    if not file.endswith(".jpg"):
                        continue
                    if training == bool(
                        re.search(
                            self.test_re, os.path.join("/", dirpath, file)
                        )
                    ):
                        continue

                    image_path = os.path.join(dirpath, file)
                    if not os.path.exists(image_path.replace(".jpg", ".npy")):
                        continue

                    yield image_path

    def _transform_example(self, image_file):
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        interpolation = (
            cv2.INTER_AREA
            if np.mean(image.shape[:2]) > self.image_size
            else cv2.INTER_CUBIC
        )
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=interpolation,
        )

        logit_file = image_file.replace(".jpg", ".npy")
        logit = np.load(logit_file)

        yield image_file, image, logit


@tf.function(jit_compile=True)
def _transform_examples(examples, preprocess):
    images = examples["image"]
    logits = examples["logit"]

    if preprocess:
        images = tf.cast(images, global_policy().compute_dtype)
        images = imagenet_utils.preprocess_input(images, mode=preprocess)

    logits = tf.reshape(logits, [tf.shape(logits)[0], 2 * logits.shape[-1]])

    return images, logits


def make_dataset(
    data_dir,
    split_name,
    batch_size,
    preprocess_mode=None,
    image_size=384,
    logits_size=1152,
):
    train_split = tfds.Split.TRAIN == split_name

    builder = Clip(
        source_dirs=[],
        data_dir=data_dir,
        image_size=image_size,
        logits_size=logits_size,
    )
    builder.download_and_prepare()

    dataset = builder.as_dataset(
        split=split_name, batch_size=None, shuffle_files=train_split
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(
        lambda examples: _transform_examples(examples, preprocess_mode),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    if train_split:
        dataset = dataset.shuffle(batch_size * 2)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
