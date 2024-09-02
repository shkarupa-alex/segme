from keras.src import backend
from keras.src import ops

from segme.ops import adjust_jpeg_quality
from segme.ops import convert_image_dtype
from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import validate


def jpeg(image, masks, weight, prob, factor, name=None):
    with backend.name_scope(name or "jpeg"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _jpeg(x, factor),
            None,
            None,
        )


def _jpeg(image, factor, name=None):
    with backend.name_scope(name or "jpeg_"):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert_image_dtype(image, "uint8")

        image_ = ops.map(lambda x: adjust_jpeg_quality(x, factor), image)
        image_.set_shape(image.shape)

        return convert_image_dtype(image_, dtype, saturate=True)
