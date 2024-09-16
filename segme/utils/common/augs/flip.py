from keras.src import backend
from keras.src import ops

from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import validate


def flip_ud(image, masks, weight, prob, name=None):
    with backend.name_scope(name or "flip_ud"):
        return apply(image, masks, weight, prob, _flip_ud, _flip_ud, _flip_ud)


def flip_lr(image, masks, weight, prob, name=None):
    with backend.name_scope(name or "flip_lr"):
        return apply(image, masks, weight, prob, _flip_lr, _flip_lr, _flip_lr)


def _flip_ud(image, name=None):
    with backend.name_scope(name or "flip_ud_"):
        image, _, _ = validate(image, None, None)

        image = ops.flip(image, axis=1)

        return image


def _flip_lr(image, name=None):
    with backend.name_scope(name or "flip_lr_"):
        image, _, _ = validate(image, None, None)

        image = ops.flip(image, axis=2)

        return image
