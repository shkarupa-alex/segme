import cv2
import numpy as np


def compose_two(fg, alpha, fg_, alpha_):
    if 3 != len(fg.shape):
        raise ValueError('Expecting `fg` rank to be 3.')

    if 3 != fg.shape[-1]:
        raise ValueError('Expecting `fg` channels size to be 3.')

    if len(alpha.shape) not in {2, 3}:
        raise ValueError('Expecting `alpha` rank to be 2 or 3.')

    if 3 == len(alpha.shape) and 1 != alpha.shape[-1]:
        raise ValueError('Expecting `alpha` channels size to be 1.')

    if 'uint8' != fg.dtype or 'uint8' != alpha.dtype:
        raise ValueError('Expecting `fg` and `alpha` dtype to be `uint8`.')

    if 3 != len(fg_.shape):
        raise ValueError('Expecting `fg_` rank to be 3.')

    if 3 != fg_.shape[-1]:
        raise ValueError('Expecting `fg_` channels size to be 3.')

    if len(alpha_.shape) not in {2, 3}:
        raise ValueError('Expecting `alpha_` rank to be 2 or 3.')

    if 3 == len(alpha_.shape) and 1 != alpha_.shape[-1]:
        raise ValueError('Expecting `alpha_` channels size to be 1.')

    if 'uint8' != fg_.dtype or 'uint8' != alpha_.dtype:
        raise ValueError('Expecting `fg_` and `alpha_` dtype to be `uint8`.')

    fg, alpha = fg.astype('float32') / 255., alpha.astype('float32') / 255.
    fg_, alpha_ = fg_.astype('float32') / 255., alpha_.astype('float32') / 255.

    # Crop meaningful parts
    mask = alpha > 0
    hindex = mask.any(1).nonzero()[0]
    top, bottom = hindex.min(), hindex.max() + 1
    windex = mask.any(0).nonzero()[0]
    left, right = windex.min(), windex.max() + 1
    fg = fg[top:bottom, left:right]
    alpha = alpha[top:bottom, left:right]

    mask_ = alpha_ > 0
    hindex_ = mask_.any(1).nonzero()[0]
    top_, bottom_ = hindex_.min(), hindex_.max() + 1
    windex_ = mask_.any(0).nonzero()[0]
    left_, right_ = windex_.min(), windex_.max() + 1
    fg_ = fg_[top_:bottom_, left_:right_]
    alpha_ = alpha_[top_:bottom_, left_:right_]

    # Resize largest to smallest
    (width, height), (width_, height_) = alpha.shape[:2], alpha_.shape[:2]
    target = min(width, width_), min(height, height_)
    alpha = cv2.resize(alpha, target, interpolation=cv2.INTER_AREA)
    alpha_ = cv2.resize(alpha_, target, interpolation=cv2.INTER_AREA)

    # Combine fgs and alphas
    # For description see https://github.com/Yaoyi-Li/GCA-Matting/issues/12
    # and https://github.com/MarcoForte/FBA_Matting/issues/44
    alpha = alpha if 3 == len(alpha.shape) else alpha[..., None]
    alpha_ = alpha_ if 3 == len(alpha_.shape) else alpha_[..., None]

    # alpha__ = 1. - (1. - alpha) * (1. - alpha_)
    # alpha__ = alpha + alpha_ - alpha * alpha_
    alpha__ = alpha + alpha_ * (1. - alpha)

    if (alpha__ == 0.).all() or (alpha__ == 1.).all():
        raise AssertionError('Composition failed')

    fg = cv2.resize(fg, target, interpolation=cv2.INTER_AREA)
    fg_ = cv2.resize(fg_, target, interpolation=cv2.INTER_AREA)

    # The overlap of two 50% transparency should be 25%
    # fg__ = fg * alpha + fg_ * (1 - alpha)
    fg__ = (fg * alpha + fg_ * alpha_ * (1 - alpha)) / (alpha__ + np.finfo(alpha__.dtype).eps)
    fg__ = np.clip(fg__, 0., 1.)

    fg__ = np.round(fg__ * 255.).astype('uint8')
    alpha__ = np.round(alpha__ * 255.).astype('uint8')

    return fg__, alpha__
