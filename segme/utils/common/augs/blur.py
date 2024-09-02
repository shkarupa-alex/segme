from keras.src import backend
from keras.src import ops

from segme.ops import convert_image_dtype
from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import validate


def gaussblur(image, masks, weight, prob, size, name=None):
    with backend.name_scope(name or "gaussblur"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _gaussblur(x, size),
            None,
            None,
        )


def _gaussblur(image, size, name=None):
    with backend.name_scope(name or "gaussblur_"):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert_image_dtype(image, "float32")

        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
        kernel = ops.cast(
            ops.arange(-size // 2 + 1, size // 2 + 1), dtype="float32"
        )
        kernel = ops.exp(
            -ops.power(kernel, 2)
            / (2.0 * ops.power(ops.cast(sigma, "float32"), 2))
        )
        kernel /= ops.sum(kernel)
        kernel = ops.tile(kernel[:, None, None], [1, image.shape[-1], 1])

        image = ops.depthwise_conv(image, kernel[None], 1, padding="same")
        image = ops.depthwise_conv(image, kernel[:, None], 1, padding="same")

        return convert_image_dtype(image, dtype, saturate=True)
