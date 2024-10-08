import numpy as np
import tensorflow as tf
from keras.src import backend
from keras.src.backend.tensorflow.core import convert_to_tensor
from keras.src.backend.tensorflow.nn import _convert_data_format
from keras.src.utils.argument_validation import standardize_tuple
from tfmiss.nn import (
    modulated_deformable_column as miss_modulated_deformable_column,
)


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


def modulated_deformable_column(
    inputs,
    offset,
    mask,
    kernel_size,
    strides,
    padding,
    dilation_rate,
    deformable_groups,
):
    return miss_modulated_deformable_column(
        inputs,
        offset,
        mask,
        kernel_size,
        strides,
        padding,
        dilation_rate,
        deformable_groups,
    )
