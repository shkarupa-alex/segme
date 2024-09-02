import numpy as np
import tensorflow as tf
from keras.src import backend
from keras.src import models
from keras.src.backend import standardize_data_format
from keras.src.backend import standardize_dtype
from keras.src.backend.tensorflow.core import cast
from keras.src.backend.tensorflow.core import convert_to_tensor
from keras.src.backend.tensorflow.nn import _convert_data_format
from keras.src.utils.argument_validation import standardize_tuple
from tensorflow.python.framework import convert_to_constants

# TODO: bias_add, linear+bias, conv+bias


def l2_normalize(x, axis=-1, epsilon=1e-12):
    x = convert_to_tensor(x)
    return tf.nn.l2_normalize(x, axis=axis, epsilon=epsilon)


def squared_difference(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    return tf.math.squared_difference(x, y)


def logdet(x):
    x = convert_to_tensor(x)
    return tf.linalg.logdet(x)


def saturate_cast(x, dtype):
    dtype = standardize_dtype(dtype)
    if isinstance(x, tf.SparseTensor):
        x_shape = x.shape
        x = tf.saturate_cast(x, dtype)
        x.set_shape(x_shape)
        return x
    else:
        return tf.saturate_cast(x, dtype)


def convert_image_dtype(x, dtype, saturate=False):  # TODO: saturate=True?
    x = backend.convert_to_tensor(x)
    dtype = tf.dtypes.as_dtype(dtype)

    if x.dtype.is_floating and dtype.is_integer:
        # TODO: https://github.com/tensorflow/tensorflow/pull/54484
        x *= dtype.max
        x = backend.numpy.round(x)

        if saturate:
            return saturate_cast(x, dtype)
        else:
            return cast(x, dtype)

    if x.dtype.is_floating and dtype.is_floating and saturate:
        return backend.numpy.clip(x, 0.0, 1.0)

    return tf.image.convert_image_dtype(x, dtype, saturate=saturate)


def fixed_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    rank = backend.numpy.ndim(kernel) - 2
    if rank not in {1, 2, 3}:
        raise ValueError(
            f"Convolution rank must be 1, 2, or 3. Got rank: {rank}."
        )

    strides = standardize_tuple(strides, rank, "strides")
    dilation_rate = standardize_tuple(dilation_rate, rank, "dilation_rate")
    if max(strides) > 1 and max(dilation_rate) > 1:
        raise ValueError(
            f"`strides > 1` not supported in conjunction with "
            f"`dilation_rate > 1`. Received: strides={strides} and "
            f"dilation_rate={dilation_rate}"
        )

    kernel_size = kernel.shape[:rank]
    if "valid" == padding or 1 == max(kernel_size) or 1 == max(strides):
        return backend.nn.conv(
            inputs,
            kernel,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )

    data_format = backend.standardize_data_format(data_format)
    tf_data_format = _convert_data_format(data_format, rank + 2)

    pad_total = [dilation_rate[i] * (kernel_size[i] - 1) for i in range(rank)]
    pad_before = [
        min(pt // 2, max(0, kernel_size[i] - strides[i]))
        for i, pt in enumerate(pad_total)
    ]

    padding = ((0, 0),) + tuple(
        (pb, pt - pb) for pt, pb in zip(pad_total, pad_before)
    )
    padding = (
        ((0, 0),) + padding
        if data_format == "channels_first"
        else padding + ((0, 0),)
    )

    return {
        1: tf.nn.conv1d,
        2: tf.nn.conv2d,
        3: tf.nn.conv3d,
    }[rank](
        inputs,
        kernel,
        strides=strides,
        padding=padding,
        dilations=dilation_rate,
        data_format=tf_data_format,
    )


def fixed_depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    rank = backend.numpy.ndim(kernel) - 2
    if rank not in {1, 2}:
        raise ValueError(f"Convolution rank must be 1 or 2. Got rank: {rank}.")

    strides = standardize_tuple(strides, rank, "strides")
    dilation_rate = standardize_tuple(dilation_rate, rank, "dilation_rate")
    if max(strides) > 1 and max(dilation_rate) > 1:
        raise ValueError(
            f"`strides > 1` not supported in conjunction with "
            f"`dilation_rate > 1`. Received: strides={strides} and "
            f"dilation_rate={dilation_rate}"
        )

    kernel_size = kernel.shape[:rank]
    if "valid" == padding or 1 == max(kernel_size) or 1 == max(strides):
        return backend.nn.depthwise_conv(
            inputs,
            kernel,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
        )

    if 1 == rank:
        strides, dilation_rate = strides[0], dilation_rate[0]
        axis = 1 if "channels_last" == data_format else 2
        inputs = tf.expand_dims(inputs, axis)
        kernel = tf.expand_dims(kernel, axis=0)

    strides = standardize_tuple(strides, 2, "strides")
    dilation_rate = standardize_tuple(dilation_rate, 2, "dilation_rate")
    kernel_size = kernel.shape[:2]

    data_format = backend.standardize_data_format(data_format)
    tf_data_format = _convert_data_format(data_format, 4)

    pad_total = [dilation_rate[i] * (kernel_size[i] - 1) for i in range(2)]
    pad_before = [
        min(pt // 2, max(0, kernel_size[i] - strides[i]))
        for i, pt in enumerate(pad_total)
    ]

    padding = ((0, 0),) + tuple(
        (pb, pt - pb) for pt, pb in zip(pad_total, pad_before)
    )
    padding = (
        padding + ((0, 0),)
        if "channels_last" == data_format
        else ((0, 0),) + padding
    )

    strides = (
        (1,) + strides + (1,)
        if "channels_last" == data_format
        else (1, 1) + strides
    )

    outputs = tf.nn.depthwise_conv2d(
        inputs,
        kernel,
        strides=strides,
        padding=padding,
        dilations=dilation_rate,
        data_format=tf_data_format,
    )

    if 1 == rank:
        outputs = tf.squeeze(outputs, [axis])

    return outputs


def adjust_brightness(x, delta):
    return tf.image.adjust_brightness(x, delta)


def adjust_contrast(x, factor):
    return tf.image.adjust_contrast(x, factor)


def adjust_gamma(x, gamma=1, gain=1):
    return tf.image.adjust_gamma(x, gamma=gamma, gain=gain)


def adjust_hue(x, delta):
    return tf.image.adjust_hue(x, delta)


def adjust_jpeg_quality(x, quality):
    return tf.image.adjust_jpeg_quality(x, quality)


def adjust_saturation(x, factor):
    return tf.image.adjust_saturation(x, factor)


def grayscale_to_rgb(x):
    return tf.image.grayscale_to_rgb(x)


def histogram_fixed_width(x, x_range, nbins=100):
    return tf.histogram_fixed_width(x, x_range, nbins=nbins)


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


def adaptive_average_pooling_2d(x, output_size):
    def case_static(x, height, width):
        start_h = (
            np.arange(output_size[0], dtype="float32") * height / output_size[0]
        )
        start_h = start_h.astype("int32")
        stop_h = (
            (np.arange(output_size[0], dtype="float32") + 1)
            * height
            / output_size[0]
        )
        stop_h = np.ceil(stop_h).astype("int32")
        size_h = stop_h - start_h
        over_h = stop_h[:-1] - start_h[1:]

        start_w = (
            np.arange(output_size[1], dtype="float32") * width / output_size[1]
        )
        start_w = start_w.astype("int32")
        stop_w = (
            (np.arange(output_size[1], dtype=np.float32) + 1)
            * width
            / output_size[1]
        )
        stop_w = np.ceil(stop_w).astype("int32")
        size_w = stop_w - start_w
        over_w = stop_w[:-1] - start_w[1:]

        kernels = np.array([size_h.max(), size_w.max()])
        if (kernels < 1).any():
            return case_static_nondivisible(x, height, width)
        if np.unique(size_h[1:-1]).size > 1 or np.unique(size_w[1:-1]).size > 1:
            return case_static_nondivisible(x, height, width)

        if np.unique(over_h).size > 1 or np.unique(over_w).size > 1:
            return case_static_nondivisible(x, height, width)
        strides = kernels - np.array([over_h.max(), over_w.max()])

        paddings = kernels - np.array([size_h.min(), size_w.min()])
        paddings_ = [
            [0, 0],
            [paddings[0], paddings[0]],
            [paddings[1], paddings[1]],
            [0, 0],
        ]

        outputs = tf.pad(x, paddings_)
        outputs = tf.nn.avg_pool(outputs, kernels, strides, "VALID")

        weights = tf.ones([1, height, width, 1], dtype=x.dtype)
        weights = tf.pad(weights, paddings_)
        weights = tf.nn.avg_pool(weights, kernels, strides, "VALID")
        outputs /= weights

        return outputs

    def case_static_nondivisible(x, height, width):
        start_h = (
            np.arange(output_size[0], dtype="float32") * height / output_size[0]
        )
        start_h = start_h.astype("int32")
        stop_h = (
            (np.arange(output_size[0], dtype="float32") + 1)
            * height
            / output_size[0]
        )
        stop_h = np.ceil(stop_h).astype("int32")

        pooled_h = []
        for idx in range(output_size[0]):
            pooled_h.append(
                tf.reduce_mean(
                    x[:, start_h[idx] : stop_h[idx]], axis=1, keepdims=True
                )
            )
        pooled_h = tf.concat(pooled_h, axis=1)

        start_w = (
            np.arange(output_size[1], dtype="float32") * width / output_size[1]
        )
        start_w = start_w.astype("int32")
        stop_w = (
            (np.arange(output_size[1], dtype=np.float32) + 1)
            * width
            / output_size[1]
        )
        stop_w = np.ceil(stop_w).astype("int32")

        pooled_w = []
        for idx in range(output_size[1]):
            pooled_w.append(
                tf.reduce_mean(
                    pooled_h[:, :, start_w[idx] : stop_w[idx]],
                    axis=2,
                    keepdims=True,
                )
            )
        pooled_w = tf.concat(pooled_w, axis=2)

        return pooled_w

    def case_dynamic_nondivisible(x, height, width):
        start_h = (
            tf.range(output_size[0], dtype="float32")
            * tf.cast(height, "float32")
            / output_size[0]
        )
        start_h = tf.cast(start_h, "int32")
        stop_h = (
            (tf.range(output_size[0], dtype="float32") + 1)
            * tf.cast(height, "float32")
            / output_size[0]
        )
        stop_h = tf.cast(tf.math.ceil(stop_h), "int32")

        pooled_h = []
        for idx in range(output_size[0]):
            pooled_h.append(
                tf.reduce_mean(
                    x[:, start_h[idx] : stop_h[idx]],
                    axis=1,
                    keepdims=True,
                )
            )
        pooled_h = tf.concat(pooled_h, axis=1)

        start_w = (
            tf.range(output_size[1], dtype="float32")
            * tf.cast(width, "float32")
            / output_size[1]
        )
        start_w = tf.cast(start_w, "int32")
        stop_w = (
            (tf.range(output_size[1], dtype="float32") + 1)
            * tf.cast(width, "float32")
            / output_size[1]
        )
        stop_w = tf.cast(tf.math.ceil(stop_w), "int32")

        pooled_w = []
        for idx in range(output_size[1]):
            pooled_w.append(
                tf.reduce_mean(
                    pooled_h[:, :, start_w[idx] : stop_w[idx]],
                    axis=2,
                    keepdims=True,
                )
            )
        pooled_w = tf.concat(pooled_w, axis=2)

        return pooled_w

    if (1, 1) == output_size:
        return tf.reduce_mean(x, axis=[1, 2], keepdims=True)

    height, width = x.shape[1:3]
    static_size = isinstance(height, int) and isinstance(width, int)
    if static_size:
        return case_static(x, height, width)

    height, width = tf.unstack(tf.shape(x)[1:3])

    return case_dynamic_nondivisible(x, height, width)


def grid_sample(
    x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
):
    x = convert_to_tensor(x)
    grid = convert_to_tensor(grid)
    if mode not in {"bilinear", "nearest"}:
        raise NotImplementedError(
            'Only "bilinear" and "nearest" interpolation methods are supported.'
        )

    if "zeros" == padding_mode:
        pad_mode = "CONSTANT"
    elif "border" == padding_mode:
        pad_mode = "SYMMETRIC"
    elif "reflection" == padding_mode:
        pad_mode = "REFLECT"
    else:
        raise ValueError("Unknown padding mode")

    features_shape = tf.shape(x)
    features_size = features_shape[1:3]
    batch_size, point_height, point_width, _ = tf.unstack(tf.shape(grid))

    assertions = [
        tf.debugging.assert_equal(
            features_shape[0],
            batch_size,
            message="Batch size should be the same for features and grid",
        ),
        tf.debugging.assert_greater_equal(
            tf.reduce_min(grid),
            tf.cast(-1.0, grid.dtype),
            message="Grid values should be in range [-1; 1]",
        ),
        tf.debugging.assert_less_equal(
            tf.reduce_max(grid),
            tf.cast(1.0, grid.dtype),
            message="Grid values should be in range [-1; 1]",
        ),
    ]
    with tf.control_dependencies(assertions):
        safe_features = tf.pad(
            x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode=pad_mode
        )
        safe_features = tf.cast(safe_features, grid.dtype)
        grid = tf.reverse(grid, axis=[-1])
        size = tf.cast(features_size, grid.dtype)

        if align_corners:
            grid = (grid + 1.0) * (size - 1) * 0.5
        else:
            grid = (grid + 1.0) * size * 0.5 - 0.5

        batch_idx = tf.reshape(tf.range(batch_size), (batch_size, 1, 1, 1))
        coord_batches = tf.tile(batch_idx, (1, point_height, point_width, 1))
        coord_bounds = tf.cast(features_size, "int32") + 1

        def _lookup(coords):
            coords = tf.clip_by_value(
                tf.cast(coords, "int32") + 1, 0, coord_bounds
            )
            indices = tf.concat([coord_batches, coords], axis=-1)
            return tf.gather_nd(safe_features, indices)

        if "bilinear" == mode:
            grid_nw = tf.math.floor(grid)
            grid_ne = grid_nw + [1, 0]
            grid_sw = grid_nw + [0, 1]
            grid_se = grid_nw + [1, 1]

            nw = tf.reduce_prod(grid_se - grid, axis=-1, keepdims=True)
            ne = tf.reduce_prod(
                (grid_sw - grid) * [1, -1], axis=-1, keepdims=True
            )
            sw = tf.reduce_prod(
                (grid_ne - grid) * [-1, 1], axis=-1, keepdims=True
            )
            se = tf.reduce_prod(grid - grid_nw, axis=-1, keepdims=True)

            result = tf.add_n(
                [
                    _lookup(grid_nw) * nw,
                    _lookup(grid_ne) * ne,
                    _lookup(grid_sw) * sw,
                    _lookup(grid_se) * se,
                ]
            )

        else:  # 'nearest' == mode:
            result = _lookup(tf.round(grid))

        features_dtype = tf.dtypes.as_dtype(x.dtype)
        if features_dtype.is_integer:
            result = tf.round(result)

        return tf.cast(result, x.dtype)


def op_type(x):
    x = convert_to_tensor(x)
    if isinstance(x, (tf.__internal__.EagerTensor, tf.Variable)):
        return None

    if not hasattr(x, "op") or not hasattr(x.op, "type"):
        return None

    return x.op.type


def model_inference_fn(model, jit_compile):
    if not isinstance(model, models.Model):
        raise ValueError(
            f"Expecting model to be an instance of `keras.Model`. "
            f"Got: {type(model)}"
        )

    input_spec = [tf.TensorSpec(i.shape, i.dtype) for i in model.inputs]
    input_spec = input_spec[0] if 1 == len(input_spec) else input_spec
    model_fn = tf.function(
        lambda x: model(x, training=False),
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    model_fn = model_fn.get_concrete_function(input_spec)
    model_fn = convert_to_constants.convert_variables_to_constants_v2(model_fn)

    return model_fn
