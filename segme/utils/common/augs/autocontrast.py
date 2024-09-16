from keras.src import backend
from keras.src import ops

from segme.ops import convert_image_dtype
from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import validate


def autocontrast(image, masks, weight, prob, name=None):
    with backend.name_scope(name or "autocontrast"):
        return apply(
            image,
            masks,
            weight,
            prob,
            _autocontrast,
            None,
            None,
        )


def _autocontrast(image, name=None):
    with backend.name_scope(name or "autocontrast_"):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert_image_dtype(image, "float32")

        lo = ops.min(image, axis=[1, 2], keepdims=True)
        hi = ops.max(image, axis=[1, 2], keepdims=True)

        image_ = ops.divide_no_nan(image - lo, hi - lo)
        image = ops.where(hi > lo, image_, image)

        return convert_image_dtype(image, dtype, saturate=True)
