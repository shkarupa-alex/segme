from keras.src import backend
from keras.src import ops

from segme.ops import convert_image_dtype
from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import blend
from segme.utils.common.augs.common import validate


def mix(image, masks, weight, prob, factor, color=None, name=None):
    with backend.name_scope(name or "mix"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _mix(x, factor, color),
            None,
            None,
        )


def _mix(image, factor, color=None, name=None):
    with backend.name_scope(name or "mix_"):
        image, _, _ = validate(image, None, None)
        factor = backend.convert_to_tensor(factor)

        batch = ops.shape(image)[0]

        if factor.shape.rank:
            factor = factor[:batch]

        if color is not None:
            color = backend.convert_to_tensor(color, image.dtype)
            color, _, _ = validate(color, None, None)
            color = color[:batch]
        else:
            color = ops.random.uniform([batch, 1, 1, image.shape[-1]])

        dtype = image.dtype
        image = convert_image_dtype(image, "float32")
        color = convert_image_dtype(color, "float32")

        image = blend(image, color, factor)

        return convert_image_dtype(image, dtype, saturate=True)
