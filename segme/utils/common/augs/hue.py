from keras.src import backend

from segme.ops import adjust_hue
from segme.ops import convert_image_dtype
from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import validate


def hue(image, masks, weight, prob, factor, name=None):
    with backend.name_scope(name or "hue"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _hue(x, factor),
            None,
            None,
        )


def _hue(image, factor, name=None):
    with backend.name_scope(name or "hue_"):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert_image_dtype(image, "float32")

        image = adjust_hue(image, factor)

        return convert_image_dtype(image, dtype, saturate=True)
