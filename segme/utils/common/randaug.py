import tensorflow as tf
from segme.utils.common.augs.common import convert, validate
from segme.utils.common.augs.autocontrast import autocontrast
from segme.utils.common.augs.blur import gaussblur
from segme.utils.common.augs.brightness import brightness
from segme.utils.common.augs.contrast import contrast
from segme.utils.common.augs.equalize import equalize
from segme.utils.common.augs.flip import flip_lr, flip_ud
from segme.utils.common.augs.gamma import gamma
from segme.utils.common.augs.grayscale import grayscale
from segme.utils.common.augs.hue import hue
from segme.utils.common.augs.invert import invert
from segme.utils.common.augs.jpeg import jpeg
from segme.utils.common.augs.mix import mix
from segme.utils.common.augs.posterize import posterize
from segme.utils.common.augs.rotate import rotate, rotate_cw, rotate_ccw
from segme.utils.common.augs.saturation import saturation
from segme.utils.common.augs.sharpness import sharpness
from segme.utils.common.augs.shear import shear_x, shear_y
from segme.utils.common.augs.shuffle import shuffle
from segme.utils.common.augs.solarize import solarize
from segme.utils.common.augs.translate import translate_x, translate_y


def _autocontrast_args(magnitude, batch, channel, replace, reduce=1.):
    prob = tf.random.uniform([batch, 1, 1, channel], maxval=magnitude / reduce)

    return [prob]


def _brightness_args(magnitude, batch, channel, replace, reduce=1., min_val=-0.5, max_val=0.5):
    min_val += 1e-5

    prob = tf.random.uniform([batch, 1, 1, 1], maxval=magnitude / reduce)
    factor = tf.random.uniform([batch, 1, 1, 1], minval=min_val * magnitude, maxval=max_val * magnitude)

    return [prob, factor]


def _contrast_args(magnitude, batch, channel, replace, reduce=1., max_val=4.):
    direction = tf.cast(tf.random.uniform([]) > 0.5, 'float32')

    prob = tf.random.uniform([batch, 1, 1, channel], maxval=magnitude / reduce)
    factor = tf.random.uniform([], minval=1., maxval=magnitude * max_val)
    factor = direction * factor + (1. - direction) / factor

    return [prob, factor]


def _equalize_args(magnitude, batch, channel, replace, reduce=1.):
    prob = tf.random.uniform([batch, 1, 1, channel], maxval=magnitude / reduce)

    return [prob]


def _flip_lr_args(magnitude, batch, channel, replace, reduce=1.):
    prob = tf.random.uniform([batch, 1, 1, 1])  # safe & efficient to apply without magnitude

    return [prob]


def _flip_ud_args(magnitude, batch, channel, replace, reduce=4.):
    prob = tf.random.uniform([batch, 1, 1, 1], maxval=magnitude / reduce)

    return [prob]


def _gamma_args(magnitude, batch, channel, replace, reduce=1., max_pow=3.):
    direction = tf.cast(tf.random.uniform([]) > 0.5, 'float32')

    prob = tf.random.uniform([batch, 1, 1, channel], maxval=magnitude / reduce)
    factor = tf.random.uniform([], minval=1., maxval=magnitude * max_pow)
    factor = direction * factor + (1. - direction) / factor

    return [prob, factor]


def _gaussblur_args(magnitude, batch, channel, replace, reduce=1.):
    prob = tf.random.uniform([batch, 1, 1, channel], maxval=magnitude / reduce)
    size = tf.random.uniform([], minval=3. - 1.49, maxval=3. + 4 * magnitude + 0.49)
    size = tf.round(size) // 2 * 2 + 1

    return [prob, size]


def _grayscale_args(magnitude, batch, channel, replace, reduce=2.):
    prob = tf.random.uniform([batch, 1, 1, channel], maxval=1 / reduce)
    factor = tf.random.uniform([batch, 1, 1, channel], maxval=magnitude)

    return [prob, factor]


def _hue_args(magnitude, batch, channel, replace, reduce=1., min_val=-0.8, max_val=0.8):
    min_val += 1e-5
    delta = (max_val - min_val) * (1. - magnitude) / 2

    prob = tf.random.uniform([batch, 1, 1, channel], maxval=magnitude / reduce)
    factor = tf.random.uniform([], minval=min_val + delta, maxval=max_val - delta)

    return [prob, factor]


def _invert_args(magnitude, batch, channel, replace, reduce=8.):
    prob = tf.random.uniform([batch, 1, 1, channel], maxval=magnitude / reduce)

    return [prob]


def _jpeg_args(magnitude, batch, channel, replace, reduce=1., min_val=30, max_val=99):
    delta = (max_val - min_val) * (1. - magnitude)

    prob = tf.random.uniform([batch, 1, 1, channel], maxval=magnitude / reduce)
    quality = tf.random.uniform([], minval=int(min_val + delta), maxval=max_val, dtype='int32')

    return [prob, quality]


def _mix_args(magnitude, batch, channel, replace, reduce=1.):
    prob = tf.random.uniform([batch, 1, 1, channel], maxval=1 / reduce)
    factor = tf.random.uniform([batch, 1, 1, channel], maxval=magnitude / 2.)

    return [prob, factor]


def _posterize_args(magnitude, batch, channel, replace, reduce=2.):
    prob = tf.random.uniform([batch, 1, 1, channel], maxval=magnitude / reduce)
    bits = tf.cast(tf.random.uniform([], minval=1, maxval=round(1 + magnitude * 7 + 1e-5), dtype='int32'), 'uint8')

    return [prob, bits]


def _rotate_args(magnitude, batch, channel, replace, reduce=1.):
    prob = tf.random.uniform([batch, 1, 1, 1], maxval=magnitude / reduce)
    degrees = tf.random.uniform([], minval=-45 * magnitude, maxval=45 * magnitude)

    return [prob, degrees, replace]


def _rotate_cw_ccw_args(magnitude, batch, channel, replace, reduce=2.):
    prob = tf.random.uniform([batch, 1, 1, 1], maxval=magnitude / reduce)

    return [prob]


def _saturation_args(magnitude, batch, channel, replace, reduce=1., max_val=4):
    direction = tf.cast(tf.random.uniform([]) > 0.5, 'float32')

    prob = tf.random.uniform([batch, 1, 1, channel], maxval=magnitude / reduce)
    factor = tf.random.uniform([], minval=1., maxval=magnitude * max_val)
    factor = direction * factor + (1. - direction) / factor

    return [prob, factor]


def _sharpness_args(magnitude, batch, channel, replace, reduce=1.):
    prob = tf.random.uniform([batch, 1, 1, channel], maxval=1 / reduce)
    factor = tf.random.uniform([batch, 1, 1, channel], maxval=magnitude / 2.)

    return [prob, factor]


def _shear_x_y_args(magnitude, batch, channel, replace, reduce=1.):
    prob = tf.random.uniform([batch, 1, 1, 1], maxval=magnitude / reduce)
    factor = tf.random.uniform([], minval=-magnitude, maxval=magnitude)

    return [prob, factor, replace]


def _shuffle_args(magnitude, batch, channel, replace, reduce=1.):
    prob = tf.random.uniform([batch, 1, 1, 1], maxval=magnitude / reduce)

    return [prob]


def _solarize_args(magnitude, batch, channel, replace, reduce=2.):
    prob = tf.random.uniform([batch, 1, 1, channel], maxval=magnitude / reduce)

    return [prob]


def _translate_x_y_args(magnitude, batch, channel, replace, reduce=1.):
    prob = tf.random.uniform([batch, 1, 1, 1], maxval=magnitude / reduce)
    factor = tf.random.uniform([], minval=-magnitude / 3, maxval=magnitude / 3)

    return [prob, factor, replace]


_AUG_FUNC = {
    'AutoContrast': autocontrast,
    'Brightness': brightness,
    'Contrast': contrast,
    'Equalize': equalize,
    'FlipLR': flip_lr,
    'FlipUD': flip_ud,
    'Gamma': gamma,
    'Gaussblur': gaussblur,
    'Grayscale': grayscale,
    'Hue': hue,
    'Invert': invert,
    'Jpeg': jpeg,
    'Mix': mix,
    'Posterize': posterize,
    'Rotate': rotate,
    'RotateCW': rotate_cw,
    'RotateCCW': rotate_ccw,
    'Saturation': saturation,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'Shuffle': shuffle,
    'Solarize': solarize,
    'TranslateX': translate_x,
    'TranslateY': translate_y
}

_AUG_ARGS = {
    'AutoContrast': _autocontrast_args,
    'Brightness': _brightness_args,
    'Contrast': _contrast_args,
    'Equalize': _equalize_args,
    'FlipLR': _flip_lr_args,
    'FlipUD': _flip_ud_args,
    'Gamma': _gamma_args,
    'Gaussblur': _gaussblur_args,
    'Grayscale': _grayscale_args,
    'Hue': _hue_args,
    'Invert': _invert_args,
    'Jpeg': _jpeg_args,
    'Mix': _mix_args,
    'Posterize': _posterize_args,
    'Rotate': _rotate_args,
    'RotateCW': _rotate_cw_ccw_args,
    'RotateCCW': _rotate_cw_ccw_args,
    'Saturation': _saturation_args,
    'Sharpness': _sharpness_args,
    'ShearX': _shear_x_y_args,
    'ShearY': _shear_x_y_args,
    'Shuffle': _shuffle_args,
    'Solarize': _solarize_args,
    'TranslateX': _translate_x_y_args,
    'TranslateY': _translate_x_y_args,
}


def rand_augment(image, masks, weight, levels=5, magnitude=0.5, ops=None, name=None):
    with tf.name_scope(name or 'rand_augment'):
        image, masks, weight = validate(image, masks, weight)
        image = convert(image, 'float32')

        if magnitude < 0. or magnitude > 1.:
            raise ValueError('Wrong magnitude value')

        batch = tf.shape(image)[0]
        channel = image.shape[-1]
        replace = tf.random.uniform([batch, 1, 1, channel])

        if ops is None:
            ops = _AUG_FUNC.keys()

        if len(ops) < levels:
            raise ValueError(
                f'Number of levels ({levels}) must be greater or equal to number of augmentations {len(ops)}.')

        selected = tf.range(0, len(ops), dtype='int32')
        selected = tf.random.shuffle(selected)
        selected = tf.unstack(selected[:levels])

        for i in range(levels):
            with tf.name_scope(f'level_{i}'):
                for j, op_name in enumerate(ops):
                    func = _AUG_FUNC[op_name]
                    args = _AUG_ARGS[op_name](magnitude, batch, channel, replace)

                    image, masks, weight = tf.cond(
                        tf.equal(selected[i], j),
                        lambda f=func, a=args: f(image, masks, weight, *a),
                        lambda: _no_op(image, masks, weight))

        image = convert(image, 'uint8')

        return image, masks, weight


def _no_op(image, masks, weight):
    _image = tf.identity(image)

    if masks is not None:
        _masks = [tf.identity(m) for m in masks]
    else:
        _masks = None

    if weight is not None:
        _weight = tf.identity(weight)
    else:
        _weight = None

    return _image, _masks, _weight
