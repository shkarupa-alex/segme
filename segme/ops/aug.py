from keras.src import backend
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation

from segme import backend as back


class AdjustBrightness(Operation):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

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

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.adjust_saturation(x, factor=self.factor)


def adjust_saturation(x, factor):
    if any_symbolic_tensors((x,)):
        return AdjustSaturation(factor=factor).symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return back.adjust_saturation(x, factor=factor)


class GrayscaleToRgb(Operation):
    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.grayscale_to_rgb(x)


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
