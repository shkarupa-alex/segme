from keras.src import backend
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.nn import Conv
from keras.src.ops.nn import DepthwiseConv
from keras.src.ops.operation import Operation

from segme import backend as back


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


class ModulatedDeformableColumn(Operation):
    def __init__(
        self, kernel_size, strides, padding, dilation_rate, deformable_groups
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.deformable_groups = deformable_groups

    # def compute_output_spec(self, x):
    #     TODO

    def call(self, inputs, offset, mask):
        inputs = backend.convert_to_tensor(inputs)
        offset = backend.convert_to_tensor(offset)
        mask = backend.convert_to_tensor(mask)
        return back.modulated_deformable_column(
            inputs,
            offset,
            mask,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            deformable_groups=self.deformable_groups,
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
    if any_symbolic_tensors((inputs, offset, mask)):
        return ModulatedDeformableColumn(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            deformable_groups=deformable_groups,
        ).symbolic_call(inputs, offset, mask)
    inputs = backend.convert_to_tensor(inputs)
    offset = backend.convert_to_tensor(offset)
    mask = backend.convert_to_tensor(mask)
    return back.modulated_deformable_column(
        inputs,
        offset,
        mask,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        deformable_groups=deformable_groups,
    )
