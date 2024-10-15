import tensorflow as tf
from keras.src import backend
from keras.src import ops
from keras.src.backend import standardize_data_format
from keras.src.backend.tensorflow.core import cast
from keras.src.backend.tensorflow.core import convert_to_tensor
from tfmiss.image import connected_components as miss_connected_components
from tfmiss.image import euclidean_distance as miss_euclidean_distance


def convert_image_dtype(x, dtype, saturate=False):  # TODO: saturate=True?
    x = backend.convert_to_tensor(x)
    dtype = tf.dtypes.as_dtype(dtype)

    if x.dtype.is_floating and dtype.is_integer:
        # TODO: https://github.com/tensorflow/tensorflow/pull/54484
        x *= dtype.max
        x = backend.numpy.round(x)

        if saturate:
            return ops.saturate_cast(x, dtype)
        else:
            return cast(x, dtype)

    if x.dtype.is_floating and dtype.is_floating and saturate:
        return backend.numpy.clip(x, 0.0, 1.0)

    return tf.image.convert_image_dtype(x, dtype, saturate=saturate)


def space_to_depth(x, block_size, data_format=None):
    x = convert_to_tensor(x)
    data_format = standardize_data_format(data_format)
    data_format_tf = "NHWC" if data_format == "channels_last" else "NCHW"
    return tf.nn.space_to_depth(x, block_size, data_format=data_format_tf)


def depth_to_space(x, block_size, data_format=None):
    x = convert_to_tensor(x)
    data_format = standardize_data_format(data_format)
    data_format_tf = "NHWC" if data_format == "channels_last" else "NCHW"
    return tf.nn.depth_to_space(x, block_size, data_format=data_format_tf)


def extract_patches(x, sizes, strides, rates, padding):
    x = convert_to_tensor(x)
    return tf.image.extract_patches(
        x,
        [1] + sizes + [1],
        [1] + strides + [1],
        [1] + rates + [1],
        padding.upper(),
    )


def dilation_2d(
    x, kernel, strides=1, padding="valid", dilations=1, data_format=None
):
    x = convert_to_tensor(x)
    kernel = convert_to_tensor(kernel)

    if isinstance(strides, int):
        strides = (strides,) * 4

    padding_tf = padding.upper()

    if isinstance(dilations, int):
        dilations = (dilations,) * 4

    data_format = standardize_data_format(data_format)
    data_format_tf = "NHWC" if data_format == "channels_last" else "NCHW"

    return tf.nn.dilation2d(
        x, kernel, strides, padding_tf, data_format_tf, dilations
    )


def erosion_2d(x, kernel, strides, padding, data_format, dilations):
    x = convert_to_tensor(x)
    kernel = convert_to_tensor(kernel)

    if isinstance(strides, int):
        strides = (strides,) * 4

    padding_tf = padding.upper()

    if isinstance(dilations, int):
        dilations = (dilations,) * 4

    data_format = standardize_data_format(data_format)
    data_format_tf = "NHWC" if data_format == "channels_last" else "NCHW"

    return tf.nn.erosion2d(
        x, kernel, strides, padding_tf, data_format_tf, dilations
    )


def connected_components(source, normalize=True):
    return miss_connected_components(source, normalize=normalize)


def euclidean_distance(source):
    return miss_euclidean_distance(source)
