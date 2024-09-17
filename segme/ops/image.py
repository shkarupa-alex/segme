from keras.src import backend
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation

from segme import backend as back


class ConvertImageDtype(Operation):
    def __init__(self, dtype, saturate=False):
        super().__init__()
        self.dtype = dtype
        self.saturate = saturate

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.convert_image_dtype(
            x, dtype=self.dtype, saturate=self.saturate
        )


def convert_image_dtype(x, dtype, saturate=False):
    if any_symbolic_tensors((x,)):
        return ConvertImageDtype(dtype=dtype, saturate=saturate).symbolic_call(
            x
        )
    x = backend.convert_to_tensor(x)
    return back.convert_image_dtype(x, dtype=dtype, saturate=saturate)


class SpaceToDepth(Operation):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.space_to_depth(x, block_size=self.block_size)


def space_to_depth(x, block_size):
    if any_symbolic_tensors((x,)):
        return SpaceToDepth(block_size=block_size).symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return back.space_to_depth(x, block_size=block_size)


class DepthToSpace(Operation):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.depth_to_space(x, block_size=self.block_size)


def depth_to_space(x, block_size):
    if any_symbolic_tensors((x,)):
        return DepthToSpace(block_size=block_size).symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return back.depth_to_space(x, block_size=block_size)


class ExtractPatches(Operation):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    # def compute_output_spec(self, x, sizes, strides, rates):
    #     TODO

    def call(self, x, sizes, strides, rates):
        x = backend.convert_to_tensor(x)
        return back.extract_patches(x, sizes, strides, rates, self.padding)


def extract_patches(x, sizes, strides, rates, padding):
    if any_symbolic_tensors((x, sizes, strides, rates)):
        return ExtractPatches(padding).symbolic_call(x, sizes, strides, rates)
    x = backend.convert_to_tensor(x)
    return back.extract_patches(x, sizes, strides, rates, padding)


class Dilation2D(Operation):
    def __init__(
        self, strides=1, padding="valid", dilations=1, data_format=None
    ):
        super().__init__()
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.data_format = data_format

    # def compute_output_spec(self, x):
    #   TODO

    def call(self, x, kernel):
        x = backend.convert_to_tensor(x)
        kernel = backend.convert_to_tensor(kernel)
        return back.dilation_2d(
            x,
            kernel,
            strides=self.strides,
            padding=self.padding,
            dilations=self.dilations,
            data_format=self.data_format,
        )


def dilation_2d(
    x, kernel, strides=1, padding="valid", dilations=1, data_format=None
):
    if any_symbolic_tensors((x, kernel)):
        return Dilation2D(
            strides=strides,
            padding=padding,
            dilations=dilations,
            data_format=data_format,
        ).symbolic_call(x, kernel)
    x = backend.convert_to_tensor(x)
    kernel = backend.convert_to_tensor(kernel)
    return back.dilation_2d(
        x,
        kernel,
        strides=strides,
        padding=padding,
        dilations=dilations,
        data_format=data_format,
    )


class Erosion2D(Operation):
    def __init__(
        self, strides=1, padding="valid", dilations=1, data_format=None
    ):
        super().__init__()
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.data_format = data_format

    # def compute_output_spec(self, x):
    #   TODO

    def call(self, x, kernel):
        x = backend.convert_to_tensor(x)
        kernel = backend.convert_to_tensor(kernel)
        return back.erosion_2d(
            x,
            kernel,
            strides=self.strides,
            padding=self.padding,
            dilations=self.dilations,
            data_format=self.data_format,
        )


def erosion_2d(
    x, kernel, strides=1, padding="valid", dilations=1, data_format=None
):
    if any_symbolic_tensors((x, kernel)):
        return Erosion2D(
            strides=strides,
            padding=padding,
            dilations=dilations,
            data_format=data_format,
        ).symbolic_call(x, kernel)
    x = backend.convert_to_tensor(x)
    kernel = backend.convert_to_tensor(kernel)
    return back.erosion_2d(
        x,
        kernel,
        strides=strides,
        padding=padding,
        dilations=dilations,
        data_format=data_format,
    )
