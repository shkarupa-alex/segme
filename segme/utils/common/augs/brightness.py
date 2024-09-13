from keras.src import backend
from keras.src import ops

from segme.ops import adjust_brightness
from segme.ops import convert_image_dtype
from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import validate


def brightness(image, masks, weight, prob, factor, name=None):
    with backend.name_scope(name or "brightness"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _brightness(x, factor),
            None,
            None,
        )


def _brightness(image, factor, name=None):
    with backend.name_scope(name or "brightness_"):
        image, _, _ = validate(image, None, None)
        factor = backend.convert_to_tensor(factor)

        if ops.ndim(factor):
            batch = ops.shape(image)[0]
            factor = factor[:batch]

        dtype = image.dtype
        image = convert_image_dtype(image, "float32")

        image = adjust_brightness(image, factor)

        return convert_image_dtype(image, dtype, saturate=True)
