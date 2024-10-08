import torch


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
