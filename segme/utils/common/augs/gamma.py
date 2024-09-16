from keras.src import backend
from keras.src import ops

from segme.ops import adjust_gamma
from segme.ops import convert_image_dtype
from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import validate


def gamma(image, masks, weight, prob, factor, invert, name=None):
    with backend.name_scope(name or "gamma"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _gamma(x, factor, invert),
            None,
            None,
        )


def _gamma(image, factor, invert, name=None):
    with backend.name_scope(name or "gamma_"):
        image, _, _ = validate(image, None, None)
        invert = backend.convert_to_tensor(invert)

        if ops.ndim(invert):
            batch = ops.shape(image)[0]
            invert = invert[:batch]

        dtype = image.dtype
        image = convert_image_dtype(image, "float32")

        image = ops.where(invert, 1.0 - image, image)
        image = adjust_gamma(image, factor)
        image = ops.where(invert, 1.0 - image, image)

        return convert_image_dtype(image, dtype, saturate=True)
