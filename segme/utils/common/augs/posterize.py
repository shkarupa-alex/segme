from keras.src import backend
from keras.src import ops

from segme.ops import convert_image_dtype
from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import validate


def posterize(image, masks, weight, prob, bits, name=None):
    with backend.name_scope(name or "posterize"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _posterize(x, bits),
            None,
            None,
        )


def _posterize(image, bits, name=None):
    with backend.name_scope(name or "posterize_"):
        image, _, _ = validate(image, None, None)
        if isinstance(bits, int) and (bits > 8 or bits < 0):
            raise ValueError("Expecting `bits` to be in range [0, 8).")

        bits = ops.cast(bits, "uint8")

        dtype = image.dtype
        image = convert_image_dtype(image, "uint8", saturate=True)

        image = ops.bitwise_right_shift(image, bits)
        image = ops.bitwise_left_shift(image, bits)

        return convert_image_dtype(image, dtype, saturate=True)
