from keras.src import backend

from segme.ops import adjust_contrast
from segme.ops import convert_image_dtype
from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import validate


def contrast(image, masks, weight, prob, factor, name=None):
    with backend.name_scope(name or "contrast"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _contrast(x, factor),
            None,
            None,
        )


def _contrast(image, factor, name=None):
    with backend.name_scope(name or "contrast_"):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert_image_dtype(image, "float32")

        image = adjust_contrast(image, factor)

        return convert_image_dtype(image, dtype, saturate=True)
