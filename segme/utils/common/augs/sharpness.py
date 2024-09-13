import numpy as np
from keras.src import backend
from keras.src import ops

from segme.ops import convert_image_dtype
from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import blend
from segme.utils.common.augs.common import validate


def sharpness(image, masks, weight, prob, factor, name=None):
    with backend.name_scope(name or "sharpness"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _sharpness(x, factor),
            None,
            None,
        )


def _sharpness(image, factor, name=None):
    with backend.name_scope(name or "sharpness_"):
        image, _, _ = validate(image, None, None)
        factor = backend.convert_to_tensor(factor)

        if ops.ndim(factor):
            batch = ops.shape(image)[0]
            factor = factor[:batch]

        dtype = image.dtype
        image = convert_image_dtype(image, "float32")

        kernel = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], "float32") / 13.0
        kernel = np.tile(kernel[..., None, None], [1, 1, 3, 1])
        kernel = ops.cast(kernel, "float32")

        image_ = ops.depthwise_conv(image, kernel, 1, padding="valid")
        image_ = ops.clip(image_, 0.0, 1.0)

        mask = ops.ones_like(image_, dtype="bool")
        mask = ops.pad(mask, [[0, 0], [1, 1], [1, 1], [0, 0]])

        image_ = ops.pad(image_, [[0, 0], [1, 1], [1, 1], [0, 0]])
        image_ = ops.where(mask, image_, image)

        image = blend(image, image_, factor)

        return convert_image_dtype(image, dtype, saturate=True)
