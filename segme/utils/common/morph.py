import numpy as np
import tensorflow as tf


def _diamond(size, dtype):
    k = np.arange(size)
    k = np.minimum(k, k[::-1])
    k = k[:, None] + k >= (size - 1) // 2

    return tf.cast(k[..., None], dtype)


def erode(inputs, size, iterations, strict=False, name=None):
    with tf.name_scope(name or "morph_erode"):
        return -dilate(-inputs, size, iterations, strict, name)


def dilate(inputs, size, iterations, strict=False, name=None):
    with tf.name_scope(name or "morph_dilate"):
        inputs = tf.convert_to_tensor(inputs)
        kernel = _diamond(size, inputs.dtype)

        if strict:
            (dilated,) = tf.while_loop(
                lambda _: True,
                lambda d: (
                    tf.nn.dilation2d(
                        d, kernel, [1] * 4, "SAME", "NHWC", [1] * 4
                    )
                    - 1,
                ),
                (inputs,),
                maximum_iterations=iterations,
            )
        else:
            (dilated,) = tf.while_loop(
                lambda _: True,
                lambda d: (
                    tf.nn.dilation2d(
                        d, kernel, [1] * 4, "SAME", "NHWC", [1] * 4
                    ),
                ),
                (inputs,),
                maximum_iterations=iterations,
            )

        return dilated
