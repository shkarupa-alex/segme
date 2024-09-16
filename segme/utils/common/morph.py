import numpy as np
from keras.src import backend
from keras.src import ops

from segme.ops import dilation_2d


def _diamond(size, dtype):
    k = np.arange(size)
    k = np.minimum(k, k[::-1])
    k = k[:, None] + k >= (size - 1) // 2

    return ops.cast(k[..., None], dtype)


def erode(inputs, size, iterations, strict=False, name=None):
    with backend.name_scope(name or "morph_erode"):
        return -dilate(-inputs, size, iterations, strict, name)


def dilate(inputs, size, iterations, strict=False, name=None):
    with backend.name_scope(name or "morph_dilate"):
        inputs = backend.convert_to_tensor(inputs)
        kernel = _diamond(size, inputs.dtype)

        if strict:
            (dilated,) = ops.while_loop(
                lambda _: True,
                lambda d: (dilation_2d(d, kernel, padding="same") - 1,),
                (inputs,),
                maximum_iterations=iterations,
            )
        else:
            (dilated,) = ops.while_loop(
                lambda _: True,
                lambda d: (dilation_2d(d, kernel, padding="same"),),
                (inputs,),
                maximum_iterations=iterations,
            )

        return dilated
