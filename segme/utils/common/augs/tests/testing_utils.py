import os
import cv2
import numpy as np
import tensorflow as tf


def aug_samples(aug_name, dtype='uint8'):
    datadir = os.path.join(os.path.dirname(__file__), 'data')
    files = [f for f in os.listdir(datadir) if '_' == f[0] and f.endswith('.jpg')]

    inputs = np.stack([cv2.imread(os.path.join(datadir, f)) for f in files])
    inputs = tf.image.convert_image_dtype(inputs, dtype)

    expected = np.stack([
        cv2.imread(os.path.join(datadir, aug_name + f.replace('.jpg', '.png'))) for f in files])
    expected = tf.image.convert_image_dtype(expected, dtype)

    return inputs, expected


def max_diff(expected, augmented):
    expected = tf.cast(expected, 'float32')
    augmented = tf.cast(augmented, 'float32')
    diff = tf.reduce_max(tf.abs(expected - augmented))

    return diff
