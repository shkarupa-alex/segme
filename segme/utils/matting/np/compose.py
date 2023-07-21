import cv2
import numpy as np
from segme.utils.matting.np.fg import solve_fg


def compose_two(fg0, alpha0, fg1, alpha1, crop=False, solve=True):
    if 3 != len(fg0.shape):
        raise ValueError('Expecting `fg` rank to be 3.')

    if 3 != fg0.shape[-1]:
        raise ValueError('Expecting `fg` channels size to be 3.')

    if len(alpha0.shape) not in {2, 3}:
        raise ValueError('Expecting `alpha` rank to be 2 or 3.')

    if 3 == len(alpha0.shape) and 1 != alpha0.shape[-1]:
        raise ValueError('Expecting `alpha` channels size to be 1.')

    if 'uint8' != fg0.dtype or 'uint8' != alpha0.dtype:
        raise ValueError('Expecting `fg` and `alpha` dtype to be `uint8`.')

    if 3 != len(fg1.shape):
        raise ValueError('Expecting `fg_` rank to be 3.')

    if 3 != fg1.shape[-1]:
        raise ValueError('Expecting `fg_` channels size to be 3.')

    if len(alpha1.shape) not in {2, 3}:
        raise ValueError('Expecting `alpha_` rank to be 2 or 3.')

    if 3 == len(alpha1.shape) and 1 != alpha1.shape[-1]:
        raise ValueError('Expecting `alpha_` channels size to be 1.')

    if 'uint8' != fg1.dtype or 'uint8' != alpha1.dtype:
        raise ValueError('Expecting `fg_` and `alpha_` dtype to be `uint8`.')

    if not isinstance(solve, (list, tuple)):
        solve = (solve,) * 2

    if crop:  # Crop meaningful parts
        mask = alpha0 > 0
        hindex = mask.any(1).nonzero()[0]
        top, bottom = hindex.min(), hindex.max() + 1
        windex = mask.any(0).nonzero()[0]
        left, right = windex.min(), windex.max() + 1
        fg0 = fg0[top:bottom, left:right]
        alpha0 = alpha0[top:bottom, left:right]

        mask_ = alpha1 > 0
        hindex_ = mask_.any(1).nonzero()[0]
        top_, bottom_ = hindex_.min(), hindex_.max() + 1
        windex_ = mask_.any(0).nonzero()[0]
        left_, right_ = windex_.min(), windex_.max() + 1
        fg1 = fg1[top_:bottom_, left_:right_]
        alpha1 = alpha1[top_:bottom_, left_:right_]

    interpolation = cv2.INTER_AREA if min(alpha1.shape[:2]) > min(alpha0.shape[:2]) else cv2.INTER_LANCZOS4

    # Resize background to be compatible with foreground
    alpha1 = cv2.resize(alpha1, alpha0.shape[1::-1], interpolation=interpolation)

    # Switch to float after resizing to beat overflow
    alpha0, alpha1 = alpha0.astype('float32') / 255., alpha1.astype('float32') / 255.

    # Combine fgs and alphas
    # For description see https://github.com/Yaoyi-Li/GCA-Matting/issues/12
    # and https://github.com/MarcoForte/FBA_Matting/issues/44
    alpha0 = alpha0 if 3 == len(alpha0.shape) else alpha0[..., None]
    alpha1 = alpha1 if 3 == len(alpha1.shape) else alpha1[..., None]
    delta = alpha1 * (1. - alpha0)
    alpha = alpha0 + delta

    if (alpha == 0.).all() or (alpha == 1.).all():
        raise AssertionError('Composition failed')

    if (alpha == alpha0).all():
        raise AssertionError('First image fully overflows second one')

    if (alpha == alpha1).all():
        raise AssertionError('Second image fully encloses first one')

    # Resize background to be compatible with foreground
    fg1 = cv2.resize(fg1, alpha0.shape[1::-1], interpolation=interpolation)

    # Switch to float after resizing to beat overflow
    fg0, fg1 = fg0.astype('float32') / 255., fg1.astype('float32') / 255.

    if solve[0]:
        fg0 = solve_fg(
            np.round(fg0 * 255.).astype('uint8'),
            np.round(alpha0 * 255.).astype('uint8')).astype('float32') / 255.

        fg1 = solve_fg(
            np.round(fg1 * 255.).astype('uint8'),
            np.round(alpha1 * 255.).astype('uint8')).astype('float32') / 255.

    # The overlap of two 50% transparency should be 25%
    fg = (fg0 * alpha0 + fg1 * delta) / (alpha + np.finfo(alpha.dtype).eps)
    fg = np.clip(fg, 0., 1.)

    fg = np.round(fg * 255.).astype('uint8')
    alpha = np.round(alpha * 255.).astype('uint8')

    if solve[1]:
        fg = solve_fg(fg, alpha)

    return fg, alpha
