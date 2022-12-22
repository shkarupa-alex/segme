import cv2
import functools
import io
import nltk
import numpy as np
import os
import resource
import tensorflow as tf
import tensorflow_datasets as tfds
from segme.policy.backbone.diy.coma import preprocess_input
from segme.utils.common import augment_onthefly


class Imagenet21k1k(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {'1.0.0': 'Initial release.'}

    CORRUPTED_FILES = {
        'n01678043_6448.JPEG', 'n01896844_997.JPEG', 'n02368116_318.JPEG', 'n02428089_710.JPEG', 'n02487347_1956.JPEG',
        'n02597972_5463.JPEG', 'n03957420_33553.JPEG', 'n03957420_30695.JPEG', 'n03957420_8296.JPEG',
        'n04135315_9318.JPEG', 'n04135315_8814.JPEG', 'n04257684_9033.JPEG', 'n04427559_2974.JPEG',
        'n06470073_47249.JPEG', 'n07930062_4147.JPEG', 'n09224725_3995.JPEG', 'n09359803_8155.JPEG',
        'n09894445_7463.JPEG', 'n12353203_3849.JPEG', 'n12630763_8018.JPEG'}

    def __init__(self, *, archive21k=None, archive1k=None, image_size=384, crop_pct=0.875, **kwargs):
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
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(self.image_size, self.image_size, 3), encoding_format='jpeg'),
                'file': tfds.features.Text(),
                'in1k': tfds.features.Scalar(dtype=tf.bool),
                'size': tfds.features.Text(),
                'label': tfds.features.Text(),
                'synset': tfds.features.Text(),
            })
        )

    def _split_generators(self, dl_manager):

        return {
            tfds.Split.VALIDATION: self._generate_examples(False),
            tfds.Split.TRAIN: self._generate_examples(True)
        }

    def _generate_examples(self, training):
        _, high = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

        skip_files = set()

        for key, features in self._generate_examples_1k(training):
            if training:
                skip_files.add(features['file'])

            yield key, features

        if training:
            for key, features in self._generate_examples_21k(skip_files):
                yield key, features

    def _generate_examples_1k(self, training):
        if self.archive1k is None:
            raise ValueError('ImageNet1k archive path must be set to generate dataset.')

        if not self.archive1k.endswith('.tar.gz'):
            raise ValueError('ImageNet1k archive has wrong extension (expected .tar.gz).')

        if not tf.io.gfile.exists(self.archive1k):
            raise ValueError(f'ImageNet1k archive {self.archive1k} is not a file.')

        for file_path, file_obj in tfds.download.iter_archive(
                self.archive1k, tfds.download.ExtractMethod.TAR_GZ_STREAM):

            if not file_path.endswith('.JPEG'):
                continue
            if training and '/train/' not in file_path:
                continue
            if not training and '/val/' not in file_path:
                continue

            if training:
                label_name = file_path.split('/train/')[-1].split('/')[0]
            else:
                label_name = self._label_from_valname(file_path)
            synset_name = self._synset_from_label(label_name)

            file_name = file_path.split('/')[-1]
            if file_name in self.CORRUPTED_FILES:
                continue

            image, size = self._prepare_image(file_obj)
            if image is None:
                continue

            yield file_name, {
                'image': image,
                'file': file_name,
                'in1k': True,
                'size': size,
                'label': label_name,
                'synset': synset_name
            }

    def _generate_examples_21k(self, skip_files):
        if self.archive21k is None:
            raise ValueError('ImageNet21k archive path must be set to generate dataset.')

        if not self.archive21k.endswith('.tar.gz'):
            raise ValueError('ImageNet21k archive has wrong extension (expected .tar.gz).')

        if not tf.io.gfile.exists(self.archive21k):
            raise ValueError(f'ImageNet21k archive {self.archive21k} is not a file.')

        for arch_path, arch_obj in tfds.download.iter_archive(
                self.archive21k, tfds.download.ExtractMethod.TAR_GZ_STREAM):

            if not arch_path.endswith('.tar'):
                continue

            label_name = arch_path.split('/')[-1].replace('.tar', '')
            synset_name = self._synset_from_label(label_name)

            for file_path, file_obj in tfds.download.iter_archive(
                    io.BytesIO(arch_obj.read()), tfds.download.ExtractMethod.TAR_STREAM):

                if not file_path.endswith('.JPEG'):
                    continue

                file_name = file_path.split('/')[-1]
                if file_name in self.CORRUPTED_FILES:
                    continue
                if file_name in skip_files:
                    continue

                image, size = self._prepare_image(file_obj)
                if image is None:
                    continue

                yield file_name, {
                    'image': image,
                    'file': file_name,
                    'in1k': False,
                    'size': size,
                    'label': label_name,
                    'synset': synset_name
                }

    @functools.lru_cache(maxsize=20000)
    def _synset_from_label(self, label):
        if not self._label_to_synset_cache:
            nltk.download('omw-1.4')
            nltk.download('wordnet')

        if label not in self._label_to_synset_cache:
            pos, index = label[0], int(label[1:].lstrip('0'))
            assert 'n' == pos, label

            synset = nltk.corpus.wordnet.synset_from_pos_and_offset(pos, index)
            self._label_to_synset_cache[label] = synset.name()

        return self._label_to_synset_cache[label]

    def _label_from_valname(self, file_path):
        if not self._val_to_label_labels:
            labels_path = tfds.image_classification.imagenet._VALIDATION_LABELS_FNAME
            labels_path = tfds.core.tfds_path(labels_path)

            with tf.io.gfile.GFile(os.fspath(labels_path)) as f:
                self._val_to_label_labels = f.read().strip().splitlines()

        assert '_val_' in file_path, file_path
        index = file_path.split('_val_')[-1].replace('.JPEG', '')
        index = int(index.lstrip('0')) - 1

        return self._val_to_label_labels[index]

    def _prepare_image(self, file_obj):
        image = np.frombuffer(file_obj.read(), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image is None:
            return None, None

        size = f'{image.shape[0]}x{image.shape[1]}'

        shape = np.array(image.shape[:2]).astype('float32')
        shape *= self.image_size / self.crop_pct / shape.min()
        shape = shape.round().astype('int32')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, shape[::-1], interpolation=cv2.INTER_CUBIC)

        pad_h, pad_w = (shape - self.image_size) // 2
        image = image[pad_h:pad_h + self.image_size, pad_w:pad_w + self.image_size]

        return image, size


@tf.function()
def _transform_examples(examples, split, mode):
    images = examples['image']
    labels = examples['synset']

    if tfds.Split.TRAIN == split:
        images = tf.image.convert_image_dtype(images, 'float32')
        images, _ = augment_onthefly(images, [])
        # TODO: https://github.com/tensorflow/tensorflow/pull/54484
        images = tf.cast(tf.round(tf.clip_by_value(images, 0., 1.) * 255.), 'uint8')

    images = preprocess_input(images)

    if 'hsmax' == mode:
        return {'images': images, 'labels': labels}

    if 'softmax' == mode:
        return images, labels

    raise ValueError(f'Unknown output mode: {mode}')


def make_dataset(data_dir, split_name, batch_size, output_mode, shuffle_files=True, drop_remainder=True):
    builder = Imagenet21k1k(data_dir=data_dir)
    builder.download_and_prepare()

    dataset = builder.as_dataset(split=split_name, batch_size=None, shuffle_files=shuffle_files)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.shuffle(batch_size * 100)
    dataset = dataset.map(
        lambda ex: _transform_examples(ex, split_name, output_mode),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset