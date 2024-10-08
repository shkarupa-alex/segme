def convert_image_dtype(x, dtype, saturate=False):  # TODO: saturate=True?
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


def connected_components(source, normalize=True):
    raise NotImplementedError


def euclidean_distance(source):
    raise NotImplementedError
