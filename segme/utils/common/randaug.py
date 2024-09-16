from keras.src import backend
from keras.src import ops

from segme.ops import convert_image_dtype
from segme.utils.common.augs.autocontrast import autocontrast
from segme.utils.common.augs.blur import gaussblur
from segme.utils.common.augs.brightness import brightness
from segme.utils.common.augs.common import validate
from segme.utils.common.augs.contrast import contrast
from segme.utils.common.augs.equalize import equalize
from segme.utils.common.augs.erase import erase
from segme.utils.common.augs.flip import flip_lr
from segme.utils.common.augs.flip import flip_ud
from segme.utils.common.augs.gamma import gamma
from segme.utils.common.augs.grayscale import grayscale
from segme.utils.common.augs.hue import hue
from segme.utils.common.augs.invert import invert
from segme.utils.common.augs.jpeg import jpeg
from segme.utils.common.augs.mix import mix
from segme.utils.common.augs.posterize import posterize
from segme.utils.common.augs.rotate import rotate
from segme.utils.common.augs.rotate import rotate_ccw
from segme.utils.common.augs.rotate import rotate_cw
from segme.utils.common.augs.saturation import saturation
from segme.utils.common.augs.sharpness import sharpness
from segme.utils.common.augs.shear import shear_x
from segme.utils.common.augs.shear import shear_y
from segme.utils.common.augs.shuffle import shuffle
from segme.utils.common.augs.solarize import solarize
from segme.utils.common.augs.translate import translate_x
from segme.utils.common.augs.translate import translate_y


def _autocontrast_args(magnitude, batch, channel, reduce=1.0):
    prob = magnitude / reduce

    return [prob]


def _brightness_args(
    magnitude, batch, channel, reduce=1.0, min_val=-0.5, max_val=0.5
):
    min_val += 1e-5

    prob = magnitude / reduce
    factor = ops.random.uniform(
        [batch, 1, 1, 1], minval=min_val * magnitude, maxval=max_val * magnitude
    )

    return [prob, factor]


def _contrast_args(magnitude, batch, channel, reduce=1.0, max_val=4.0):
    direction = ops.cast(ops.random.uniform([]) > 0.5, "float32")

    prob = magnitude / reduce
    factor = ops.random.uniform([], minval=1.0, maxval=magnitude * max_val)
    factor = direction * factor + (1.0 - direction) / factor

    return [prob, factor]


def _equalize_args(magnitude, batch, channel, reduce=1.0):
    prob = magnitude / reduce

    return [prob]


def _erase_args(
    magnitude, batch, channel, reduce=1.0, min_area=0.02, max_area=0.5
):
    prob = magnitude / reduce
    area = (min_area, max_area * magnitude)

    return [prob, area]


def _flip_lr_args(magnitude, batch, channel, reduce=1.0):
    prob = 1.0  # safe & efficient to apply without magnitude

    return [prob]


def _flip_ud_args(magnitude, batch, channel, reduce=4.0):
    prob = magnitude / reduce

    return [prob]


def _gamma_args(magnitude, batch, channel, reduce=1.0, max_pow=3.0):
    direction = ops.cast(ops.random.uniform([]) > 0.5, "float32")

    prob = magnitude / reduce
    factor = ops.random.uniform([], minval=1.0, maxval=magnitude * max_pow)
    factor = direction * factor + (1.0 - direction) / factor

    invert = ops.random.uniform([batch, 1, 1, 1]) > 0.5

    return [prob, factor, invert]


def _gaussblur_args(magnitude, batch, channel, reduce=1.0):
    prob = magnitude / reduce
    size = ops.random.uniform(
        [], minval=3.0 - 1.49, maxval=3.0 + 4 * magnitude + 0.49
    )
    size = ops.round(size) // 2 * 2 + 1

    return [prob, size]


def _grayscale_args(magnitude, batch, channel, reduce=2.0):
    prob = magnitude / reduce
    factor = ops.random.uniform([batch, 1, 1, channel], maxval=magnitude)

    return [prob, factor]


def _hue_args(magnitude, batch, channel, reduce=1.0, min_val=-0.8, max_val=0.8):
    min_val += 1e-5
    delta = (max_val - min_val) * (1.0 - magnitude) / 2

    prob = magnitude / reduce
    factor = ops.random.uniform(
        [], minval=min_val + delta, maxval=max_val - delta
    )

    return [prob, factor]


def _invert_args(magnitude, batch, channel, reduce=12.0):
    prob = magnitude / reduce

    return [prob]


def _jpeg_args(magnitude, batch, channel, reduce=1.0, min_val=30, max_val=99):
    delta = (max_val - min_val) * (1.0 - magnitude)

    prob = magnitude / reduce
    quality = ops.random.uniform(
        [], minval=int(min_val + delta), maxval=max_val, dtype="int32"
    )

    return [prob, quality]


def _mix_args(magnitude, batch, channel, reduce=1.0):
    prob = magnitude / reduce
    factor = ops.random.uniform([batch, 1, 1, channel], maxval=magnitude / 2.0)

    return [prob, factor]


def _posterize_args(magnitude, batch, channel, reduce=2.0):
    prob = magnitude / reduce
    bits = ops.cast(
        ops.random.uniform(
            [], minval=1, maxval=round(1 + magnitude * 6 + 1e-5), dtype="int32"
        ),
        "uint8",
    )

    return [prob, bits]


def _rotate_args(magnitude, batch, channel, reduce=1.0):
    prob = magnitude / reduce
    degrees = ops.random.uniform(
        [], minval=-45 * magnitude, maxval=45 * magnitude
    )

    return [prob, degrees]


def _rotate_cw_ccw_args(magnitude, batch, channel, reduce=2.0):
    prob = magnitude / reduce

    return [prob]


def _saturation_args(magnitude, batch, channel, reduce=1.0, max_val=4):
    direction = ops.cast(ops.random.uniform([]) > 0.5, "float32")

    prob = magnitude / reduce
    factor = ops.random.uniform([], minval=1.0, maxval=magnitude * max_val)
    factor = direction * factor + (1.0 - direction) / factor

    return [prob, factor]


def _sharpness_args(magnitude, batch, channel, reduce=1.0):
    prob = magnitude / reduce
    factor = ops.random.uniform([batch, 1, 1, channel], maxval=magnitude / 2.0)

    return [prob, factor]


def _shear_x_y_args(magnitude, batch, channel, reduce=1.0):
    prob = magnitude / reduce
    factor = ops.random.uniform([], minval=-magnitude, maxval=magnitude)

    return [prob, factor]


def _shuffle_args(magnitude, batch, channel, reduce=1.0):
    prob = magnitude / reduce

    return [prob]


def _solarize_args(magnitude, batch, channel, reduce=4.0):
    prob = magnitude / reduce

    return [prob]


def _translate_x_y_args(magnitude, batch, channel, reduce=1.0):
    prob = magnitude / reduce
    factor = ops.random.uniform([], minval=-magnitude / 3, maxval=magnitude / 3)

    return [prob, factor]


_AUG_FUNC = {
    "AutoContrast": autocontrast,
    "Brightness": brightness,
    "Contrast": contrast,
    "Equalize": equalize,
    "Erase": erase,
    "FlipLR": flip_lr,
    "FlipUD": flip_ud,
    "Gamma": gamma,
    "Gaussblur": gaussblur,
    "Grayscale": grayscale,
    "Hue": hue,
    "Invert": invert,
    "Jpeg": jpeg,
    "Mix": mix,
    "Posterize": posterize,
    "Rotate": rotate,
    "RotateCCW": rotate_ccw,
    "RotateCW": rotate_cw,
    "Saturation": saturation,
    "Sharpness": sharpness,
    "ShearX": shear_x,
    "ShearY": shear_y,
    "Shuffle": shuffle,
    "Solarize": solarize,
    "TranslateX": translate_x,
    "TranslateY": translate_y,
}

_AUG_ARGS = {
    "AutoContrast": _autocontrast_args,
    "Brightness": _brightness_args,
    "Contrast": _contrast_args,
    "Equalize": _equalize_args,
    "Erase": _erase_args,
    "FlipLR": _flip_lr_args,
    "FlipUD": _flip_ud_args,
    "Gamma": _gamma_args,
    "Gaussblur": _gaussblur_args,
    "Grayscale": _grayscale_args,
    "Hue": _hue_args,
    "Invert": _invert_args,
    "Jpeg": _jpeg_args,
    "Mix": _mix_args,
    "Posterize": _posterize_args,
    "Rotate": _rotate_args,
    "RotateCCW": _rotate_cw_ccw_args,
    "RotateCW": _rotate_cw_ccw_args,
    "Saturation": _saturation_args,
    "Sharpness": _sharpness_args,
    "ShearX": _shear_x_y_args,
    "ShearY": _shear_x_y_args,
    "Shuffle": _shuffle_args,
    "Solarize": _solarize_args,
    "TranslateX": _translate_x_y_args,
    "TranslateY": _translate_x_y_args,
}


def _no_op(image, masks, weight):
    _image = image

    if masks is not None:
        _masks = [m for m in masks]
    else:
        _masks = None

    if weight is not None:
        _weight = weight
    else:
        _weight = None

    return _image, _masks, _weight


def rand_augment_full(
    image, masks, weight, levels=5, magnitude=0.5, operations=None, name=None
):
    with backend.name_scope(name or "rand_augment"):
        image, masks, weight = validate(image, masks, weight)

        if 0 == levels or 0.0 == magnitude:
            return image, masks, weight

        if magnitude < 0.0 or magnitude > 1.0:
            raise ValueError("Wrong magnitude value")

        if operations is None:
            operations = list(_AUG_FUNC.keys())

        if len(operations) < levels:
            raise ValueError(
                f"Number of levels ({levels}) must be greater or "
                f"equal to number of augmentations {len(operations)}."
            )

        dtype = image.dtype
        image = convert_image_dtype(image, "float32")

        batch, _, _, channel = ops.shape(image)

        selected = ops.arange(len(operations), dtype="int32")
        selected = ops.random.shuffle(selected)
        selected = ops.unstack(selected[:levels])

        for i in range(levels):
            with backend.name_scope(f"level_{i}"):
                for j, op_name in enumerate(operations):
                    func = _AUG_FUNC[op_name]
                    args = _AUG_ARGS[op_name](magnitude, batch, channel)

                    image, masks, weight = ops.cond(
                        ops.equal(selected[i], j),
                        lambda f=func, a=args: f(image, masks, weight, *a),
                        lambda: _no_op(image, masks, weight),
                    )

        image = convert_image_dtype(image, dtype)

        return image, masks, weight


def rand_augment_safe(
    image, masks, weight, levels=5, magnitude=0.5, operations=None, name=None
):
    if operations is None:
        operations = list(_AUG_FUNC.keys())

    operations = list(
        set(operations)
        - {
            "Erase",
            "Invert",
            "Rotate",
            "ShearX",
            "ShearY",
            "TranslateX",
            "TranslateY",
        }
    )

    return rand_augment_full(
        image,
        masks,
        weight,
        levels=levels,
        magnitude=magnitude,
        operations=operations,
        name=name,
    )


def rand_augment_matting(
    image, masks, weight, levels=5, magnitude=0.4, operations=None, name=None
):
    if operations is None:
        operations = list(_AUG_FUNC.keys())

    operations = list(
        set(operations)
        - {
            "Erase",
            "Equalize",
            "Gaussblur",
            "Grayscale",
            "Invert",
            "Posterize",
            "Rotate",
            "Sharpness",
            "ShearX",
            "ShearY",
            "Solarize",
            "TranslateX",
            "TranslateY",
        }
    )

    return rand_augment_full(
        image,
        masks,
        weight,
        levels=levels,
        magnitude=magnitude,
        operations=operations,
        name=name,
    )
