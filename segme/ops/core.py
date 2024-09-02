from keras.src import backend
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.nn import Conv
from keras.src.ops.nn import DepthwiseConv
from keras.src.ops.operation import Operation

from segme import backend as back


class L2Normalize(Operation):
    def __init__(self, axis=-1, epsilon=1e-12):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon

    def compute_output_spec(self, x):
        output_dtype = backend.standardize_dtype(x.dtype)
        if "int" in output_dtype or output_dtype == "bool":
            output_dtype = backend.floatx()
        backend.KerasTensor(shape=x.shape, dtype=output_dtype)

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.l2_normalize(x, axis=self.axis, epsilon=self.epsilon)


def l2_normalize(x, axis=-1, epsilon=1e-12):
    """Normalizes along dimension axis using an L2 norm.

    Args:
        x: Input tensor.
        axis: Dimension along which to normalize. Defaults to -1.
        epsilon: Small float added to variance to avoid dividing by zero.
            Defaults to 1e-12.

    Returns:
        L2-normalized tensor.
    """
    if any_symbolic_tensors((x,)):
        return L2Normalize(axis=axis, epsilon=epsilon).symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return back.l2_normalize(x, axis=axis, epsilon=epsilon)


class SquaredDifference(Operation):
    # def compute_output_spec(self, x):
    #   TODO

    def call(self, x, y):
        x = backend.convert_to_tensor(x)
        y = backend.convert_to_tensor(y)
        return back.squared_difference(x, y)


def squared_difference(x, y):
    if any_symbolic_tensors((x, y)):
        return SquaredDifference().symbolic_call(x, y)
    x = backend.convert_to_tensor(x)
    y = backend.convert_to_tensor(y)
    return back.squared_difference(x, y)


class Logdet(Operation):
    # def compute_output_spec(self, x):
    #   TODO

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.logdet(x)


def logdet(x):
    if any_symbolic_tensors((x,)):
        return Logdet().symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return back.logdet(x)


class SaturateCast(Operation):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def compute_output_spec(self, x):
        return backend.KerasTensor(shape=x.shape, dtype=self.dtype)

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.saturate_cast(x, dtype=self.dtype)


def saturate_cast(x, dtype):
    if any_symbolic_tensors((x,)):
        return SaturateCast(dtype=dtype).symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return back.saturate_cast(x, dtype=dtype)


class ConvertImageDtype(Operation):
    def __init__(self, dtype, saturate=False):
        super().__init__()
        self.dtype = dtype
        self.saturate = saturate

    # def compute_output_spec(self, x):
    #    TODO
    #     return backend.KerasTensor(shape=x.shape, dtype=self.dtype)

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


class FixedConv(Conv):
    def call(self, inputs, kernel):
        return back.fixed_conv(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )


def fixed_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = backend.standardize_data_format(data_format)
    padding = padding.lower()
    if any_symbolic_tensors((inputs,)):
        return FixedConv(
            strides, padding, data_format, dilation_rate
        ).symbolic_call(inputs, kernel)
    return back.fixed_conv(
        inputs, kernel, strides, padding, data_format, dilation_rate
    )


class FixedDepthwiseConv(DepthwiseConv):
    def call(self, inputs, kernel):
        return back.fixed_depthwise_conv(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )


def fixed_depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = backend.standardize_data_format(data_format)
    padding = padding.lower()
    if any_symbolic_tensors((inputs,)):
        return FixedDepthwiseConv(
            strides, padding, data_format, dilation_rate
        ).symbolic_call(inputs, kernel)
    return back.fixed_depthwise_conv(
        inputs, kernel, strides, padding, data_format, dilation_rate
    )


class AdjustBrightness(Operation):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    # def compute_output_spec(self, x):
    #    TODO
    #     return backend.KerasTensor(shape=x.shape, dtype=self.dtype)

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.adjust_brightness(x, delta=self.delta)


def adjust_brightness(x, delta):
    if any_symbolic_tensors((x,)):
        return AdjustBrightness(delta=delta).symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return back.adjust_brightness(x, delta=delta)


class AdjustContrast(Operation):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    # def compute_output_spec(self, x):
    #    TODO
    #     return backend.KerasTensor(shape=x.shape, dtype=self.dtype)

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.adjust_contrast(x, factor=self.factor)


def adjust_contrast(x, factor):
    if any_symbolic_tensors((x,)):
        return AdjustContrast(factor=factor).symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return back.adjust_contrast(x, factor=factor)


class AdjustGamma(Operation):
    def __init__(self, gamma=1, gain=1):
        super().__init__()
        self.gamma = gamma
        self.gain = gain

    # def compute_output_spec(self, x):
    #    TODO
    #     return backend.KerasTensor(shape=x.shape, dtype=self.dtype)

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.adjust_gamma(x, gamma=self.gamma, gain=self.gain)


def adjust_gamma(x, gamma=1, gain=1):
    if any_symbolic_tensors((x,)):
        return AdjustGamma(gamma=gamma, gain=gain).symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return back.adjust_gamma(x, gamma=gamma, gain=gain)


class AdjustHue(Operation):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    # def compute_output_spec(self, x):
    #    TODO
    #     return backend.KerasTensor(shape=x.shape, dtype=self.dtype)

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.adjust_hue(x, delta=self.delta)


def adjust_hue(x, delta):
    if any_symbolic_tensors((x,)):
        return AdjustHue(delta=delta).symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return back.adjust_hue(x, delta=delta)


class AdjustJpegQuality(Operation):
    def __init__(self, quality):
        super().__init__()
        self.quality = quality

    # def compute_output_spec(self, x):
    #    TODO
    #     return backend.KerasTensor(shape=x.shape, dtype=self.dtype)

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.adjust_jpeg_quality(x, quality=self.quality)


def adjust_jpeg_quality(x, quality):
    if any_symbolic_tensors((x,)):
        return AdjustJpegQuality(quality=quality).symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return back.adjust_jpeg_quality(x, quality=quality)


class AdjustSaturation(Operation):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    # def compute_output_spec(self, x):
    #    TODO
    #     return backend.KerasTensor(shape=x.shape, dtype=self.dtype)

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.adjust_saturation(x, factor=self.factor)


def adjust_saturation(x, factor):
    if any_symbolic_tensors((x,)):
        return AdjustSaturation(factor=factor).symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return back.adjust_saturation(x, factor=factor)


class GrayscaleToRgb(Operation):
    # def compute_output_spec(self, x):
    #    TODO
    #     return backend.KerasTensor(shape=x.shape, dtype=self.dtype)

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.GrayscaleToRgb(x)


def grayscale_to_rgb(x):
    if any_symbolic_tensors((x,)):
        return GrayscaleToRgb().symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return back.grayscale_to_rgb(x)


class HistogramFixedWidth(Operation):
    def __init__(self, x_range, nbins=100):
        super().__init__()
        self.x_range = x_range
        self.nbins = nbins

    # def compute_output_spec(self, x):
    #    TODO
    #     return backend.KerasTensor(shape=x.shape, dtype=self.dtype)

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.histogram_fixed_width(
            x, x_range=self.x_range, nbins=self.nbins
        )


def histogram_fixed_width(x, x_range, nbins=100):
    if any_symbolic_tensors((x,)):
        return HistogramFixedWidth(x_range=x_range, nbins=nbins).symbolic_call(
            x
        )
    x = backend.convert_to_tensor(x)
    return back.histogram_fixed_width(x, x_range=x_range, nbins=nbins)


class SpaceToDepth(Operation):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    # def compute_output_spec(self, x):
    #     TODO

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

    # def compute_output_spec(self, x):
    #     TODO

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


class AdaptiveAveragePooling2D(Operation):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def compute_output_spec(self, x):
        return KerasTensor(
            shape=(
                x.shape[0],
                self.output_size[0],
                self.output_size[1],
                x.shape[3],
            ),
            dtype=x.dtype,
        )

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.adaptive_average_pooling_2d(x, output_size=self.output_size)


def adaptive_average_pooling_2d(x, output_size):
    if any_symbolic_tensors((x,)):
        return AdaptiveAveragePooling2D(output_size=output_size).symbolic_call(
            x
        )
    x = backend.convert_to_tensor(x)
    return back.adaptive_average_pooling_2d(x, output_size=output_size)


class GridSample(Operation):
    def __init__(
        self, mode="bilinear", padding_mode="zeros", align_corners=False
    ):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    # def compute_output_spec(self, x):
    #     TODO

    def call(self, x, grid):
        x = backend.convert_to_tensor(x)
        grid = backend.convert_to_tensor(grid)
        return back.grid_sample(
            x,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


def grid_sample(
    x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
):
    if any_symbolic_tensors((x, grid)):
        return GridSample(
            mode=mode, padding_mode=padding_mode, align_corners=align_corners
        ).symbolic_call(x, grid)
    x = backend.convert_to_tensor(x)
    return back.grid_sample(
        x,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
