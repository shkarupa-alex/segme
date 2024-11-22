import os

import cv2
import numpy as np
from keras.src import ops

from segme.ops import convert_image_dtype


def aug_samples(aug_name, dtype="uint8"):
    datadir = os.path.join(os.path.dirname(__file__), "data")
    files = [
        f for f in os.listdir(datadir) if "_" == f[0] and f.endswith(".jpg")
    ]
    files = sorted(files)

    inputs = np.stack([cv2.imread(os.path.join(datadir, f)) for f in files])
    inputs = convert_image_dtype(inputs, dtype)

    expected = np.stack(
        [
            cv2.imread(
                os.path.join(datadir, aug_name + f.replace(".jpg", ".png"))
            )
            for f in files
        ]
    )
    expected = convert_image_dtype(expected, dtype)

    return inputs, expected


def max_diff(expected, augmented, axis=None):
    expected = ops.cast(expected, "float32")
    augmented = ops.cast(augmented, "float32")
    diff = ops.max(ops.abs(expected - augmented), axis=axis)

    return diff
