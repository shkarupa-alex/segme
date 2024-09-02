import torch
from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.torch.core import cast
from keras.src.backend.torch.core import convert_to_tensor


def l2_normalize(x, axis=-1, epsilon=1e-12):
    return torch.nn.functional.normalize(x, p=2.0, dim=axis, eps=epsilon)


def squared_difference(x, y):
    raise NotImplementedError


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
    raise NotImplementedError  # TODO fold?


def dilation_2d(
    x, kernel, strides=1, padding="valid", dilations=1, data_format=None
):
    raise NotImplementedError


def erosion_2d(x, kernel, strides, padding, data_format, dilations):
    raise NotImplementedError


def adaptive_average_pooling_2d(x, size):
    return torch.nn.functional.adaptive_avg_pool2d(x, size)


def grid_sample(
    x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
):
    return torch.nn.functional.grid_sample(
        x,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def op_type(x):
    return None


def model_inference_fn(model, jit_compile):
    raise NotImplementedError
