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


def adaptive_average_pooling_2d(x, size):
    raise NotImplementedError


def grid_sample(
    x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
):
    raise NotImplementedError


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
    raise NotImplementedError
