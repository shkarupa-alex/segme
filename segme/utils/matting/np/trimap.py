import cv2
import numpy as np


def alpha_trimap(alpha, size):
    if len(alpha.shape) not in {2, 3}:
        raise ValueError("Expecting `alpha` rank to be 2 or 3.")

    if 3 == len(alpha.shape) and 1 != alpha.shape[-1]:
        raise ValueError("Expecting `alpha` channels size to be 1.")

    if "uint8" != alpha.dtype:
        raise ValueError("Expecting `alpha` dtype to be `uint8`.")

    if isinstance(size, tuple) and 2 == len(size):
        iterations = np.random.uniform(size[0], size[1] + 1, 2).astype("int32")
    elif isinstance(size, int):
        iterations = np.array([size, size])
    else:
        raise ValueError(
            "Expecting `size` to be a single margin or a tuple of [min; max] margins."
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(alpha, kernel, iterations=iterations[0])
    eroded = cv2.erode(alpha, kernel, iterations=iterations[1])

    trimap = np.full(alpha.shape, 128, dtype=alpha.dtype)
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0

    return trimap
