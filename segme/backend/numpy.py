import numpy as np


def l2_normalize(x, axis=-1, epsilon=1e-12):
    square_sum = np.sum(np.square(x), axis=axis, keepdims=True)
    x_norm = np.sqrt(np.maximum(square_sum, epsilon))

    return x / x_norm


def logdet(x):
    raise NotImplementedError


def saturate_cast(x, dtype):
    raise NotImplementedError


def convert_image_dtype(x, dtype, saturate=False):  # TODO: saturate=True?
    raise NotImplementedError


def fixed_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    raise NotImplementedError


def fixed_depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    raise NotImplementedError


def adjust_brightness(x, delta):
    raise NotImplementedError


def adjust_contrast(x, factor):
    raise NotImplementedError


def adjust_gamma(x, gamma=1, gain=1):
    raise NotImplementedError


def adjust_hue(x, delta):
    raise NotImplementedError


def adjust_jpeg_quality(x, quality):
    raise NotImplementedError


def adjust_saturation(x, factor):
    raise NotImplementedError


def grayscale_to_rgb(x):
    raise NotImplementedError


def histogram_fixed_width(x, x_range, nbins=100):
    raise NotImplementedError


def space_to_depth(x, block_size, data_format=None):
    raise NotImplementedError


def depth_to_space(x, block_size, data_format=None):
    raise NotImplementedError


def extract_patches(x, sizes, strides, rates, padding):
    raise NotImplementedError


def dilation_2d(
    x, kernel, strides=1, padding="valid", dilations=1, data_format=None
):
    raise NotImplementedError


def erosion_2d(x, kernel, strides, padding, data_format, dilations):
    raise NotImplementedError


def adaptive_average_pooling_2d(x, size):
    raise NotImplementedError


def grid_sample(
    x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
):
    raise NotImplementedError


def op_type(x):
    return None


def model_inference_fn(model, jit_compile):
    raise NotImplementedError
