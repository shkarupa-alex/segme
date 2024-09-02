from keras.src import backend

from segme.ops import adjust_saturation
from segme.ops import convert_image_dtype
from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import validate


def saturation(image, masks, weight, prob, factor, name=None):
    with backend.name_scope(name or "saturation"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _saturation(x, factor),
            None,
            None,
        )


def _saturation(image, factor, name=None):
    with backend.name_scope(name or "saturation_"):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert_image_dtype(image, "float32")

        image = adjust_saturation(image, factor)

        return convert_image_dtype(image, dtype, saturate=True)
