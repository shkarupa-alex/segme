from keras.src import backend
from keras.src import ops

from segme.ops import convert_image_dtype
from segme.ops import grayscale_to_rgb
from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import blend
from segme.utils.common.augs.common import validate


def grayscale(image, masks, weight, prob, factor, name=None):
    with backend.name_scope(name or "grayscale"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _grayscale(x, factor),
            None,
            None,
        )


def _grayscale(image, factor, name=None):
    with backend.name_scope(name or "grayscale_"):
        image, _, _ = validate(image, None, None)
        factor = backend.convert_to_tensor(factor)

        if factor.shape.rank:
            batch = ops.shape(image)[0]
            factor = factor[:batch]

        dtype = image.dtype
        image = convert_image_dtype(image, "float32")

        image_ = ops.image.rgb_to_grayscale(image)
        image_ = grayscale_to_rgb(image_)
        image = blend(image, image_, factor)

        return convert_image_dtype(image, dtype, saturate=True)
