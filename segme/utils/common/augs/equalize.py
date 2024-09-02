from keras.src import backend
from keras.src import ops

from segme.ops import convert_image_dtype
from segme.ops import histogram_fixed_width
from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import validate


def equalize(image, masks, weight, prob, name=None):
    with backend.name_scope(name or "equalize"):
        return apply(image, masks, weight, prob, _equalize, None, None)


def _equalize(image, name=None):
    with backend.name_scope(name or "equalize_"):
        image, _, _ = validate(image, None, None)

        dtype = image.dtype
        image = convert_image_dtype(image, "uint8", saturate=True)

        def _equalize_2d(image):
            histo = histogram_fixed_width(image, [0, 255], nbins=256)

            step = histo[histo > 0]
            step = (ops.sum(step) - step[-1]) // 255

            lut = (ops.cumsum(histo) + (step // 2)) // ops.maximum(step, 1)
            lut = ops.concatenate([ops.zeros((1,), "int32"), lut[:-1]], 0)
            lut = ops.clip(lut, 0, 255)

            image_ = ops.take(lut, image, axis=0)

            apply = ops.cast(step > 0, image.dtype)
            image = image_ * apply + image * (1 - apply)

            return image

        batch, height, width, channel = ops.shape(image)
        image_ = ops.cast(image, "int32")
        image_ = ops.transpose(image_, [0, 3, 1, 2])
        image_ = ops.reshape(image_, [batch * channel, height, width])
        image_ = ops.map(_equalize_2d, image_)
        image_ = ops.reshape(image_, [batch, channel, height, width])
        image_ = ops.transpose(image_, [0, 2, 3, 1])
        image_.set_shape(image.shape)
        image = ops.cast(image_, "uint8")

        return convert_image_dtype(image, dtype, saturate=True)
