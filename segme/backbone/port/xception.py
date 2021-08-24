from functools import partial
from keras import models
from ..utils import patch_config, wrap_bone
from . import aligned_xception

AlignedXception41 = partial(
    wrap_bone,
    aligned_xception.Xception41,
    aligned_xception.preprocess_input)


def AlignedXception41Stride16(*args, **kwargs):
    base = AlignedXception41(*args, **kwargs)
    conf = base.get_config()

    conf = patch_config(conf, ['exit_flow/block1/unit1/sepconv3_depthwise'], 'strides', (1, 1))
    conf = patch_config(conf, ['exit_flow/block1/unit1/shortcut'], 'strides', (1, 1))
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv1_depthwise'], 'dilation_rate', (2, 2))
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv2_depthwise'], 'dilation_rate', (2, 2))
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv3_depthwise'], 'dilation_rate', (2, 2))

    patch = models.Model.from_config(conf)
    patch.set_weights(base.get_weights())

    return patch


def AlignedXception41Stride8(*args, **kwargs):
    base = AlignedXception41(*args, **kwargs)
    conf = base.get_config()

    conf = patch_config(conf, ['entry_flow/block3/unit1/sepconv3_depthwise'], 'strides', (1, 1))
    conf = patch_config(conf, ['entry_flow/block3/unit1/shortcut'], 'strides', (1, 1))

    for i in range(8):
        conf = patch_config(conf, [
            'middle_flow/block1/unit{}/sepconv1_depthwise'.format(i + 1)], 'dilation_rate', (2, 2))
        conf = patch_config(conf, [
            'middle_flow/block1/unit{}/sepconv2_depthwise'.format(i + 1)], 'dilation_rate', (2, 2))
        conf = patch_config(conf, [
            'middle_flow/block1/unit{}/sepconv3_depthwise'.format(i + 1)], 'dilation_rate', (2, 2))

    conf = patch_config(conf, ['exit_flow/block1/unit1/sepconv3_depthwise'], 'strides', (1, 1))
    conf = patch_config(conf, ['exit_flow/block1/unit1/shortcut'], 'strides', (1, 1))

    # Output shape is different when using exit dilation rates like in
    # https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py#L265
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv1_depthwise'], 'dilation_rate', (2, 2))
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv2_depthwise'], 'dilation_rate', (2, 2))
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv3_depthwise'], 'dilation_rate', (2, 2))

    patch = models.Model.from_config(conf)
    patch.set_weights(base.get_weights())

    return patch


AlignedXception65 = partial(
    wrap_bone,
    aligned_xception.Xception65,
    aligned_xception.preprocess_input)


def AlignedXception65Stride16(*args, **kwargs):
    base = AlignedXception65(*args, **kwargs)
    conf = base.get_config()

    conf = patch_config(conf, ['exit_flow/block1/unit1/sepconv3_depthwise'], 'strides', (1, 1))
    conf = patch_config(conf, ['exit_flow/block1/unit1/shortcut'], 'strides', (1, 1))
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv1_depthwise'], 'dilation_rate', (2, 2))
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv2_depthwise'], 'dilation_rate', (2, 2))
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv3_depthwise'], 'dilation_rate', (2, 2))

    patch = models.Model.from_config(conf)
    patch.set_weights(base.get_weights())

    return patch


def AlignedXception65Stride8(*args, **kwargs):
    base = AlignedXception65(*args, **kwargs)
    conf = base.get_config()

    conf = patch_config(conf, ['entry_flow/block3/unit1/sepconv3_depthwise'], 'strides', (1, 1))
    conf = patch_config(conf, ['entry_flow/block3/unit1/shortcut'], 'strides', (1, 1))

    for i in range(16):
        conf = patch_config(conf, [
            'middle_flow/block1/unit{}/sepconv1_depthwise'.format(i + 1)], 'dilation_rate', (2, 2))
        conf = patch_config(conf, [
            'middle_flow/block1/unit{}/sepconv2_depthwise'.format(i + 1)], 'dilation_rate', (2, 2))
        conf = patch_config(conf, [
            'middle_flow/block1/unit{}/sepconv3_depthwise'.format(i + 1)], 'dilation_rate', (2, 2))

    conf = patch_config(conf, ['exit_flow/block1/unit1/sepconv3_depthwise'], 'strides', (1, 1))
    conf = patch_config(conf, ['exit_flow/block1/unit1/shortcut'], 'strides', (1, 1))

    # Output shape is different when using exit dilation rates like in
    # https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py#L265
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv1_depthwise'], 'dilation_rate', (2, 2))
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv2_depthwise'], 'dilation_rate', (2, 2))
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv3_depthwise'], 'dilation_rate', (2, 2))

    patch = models.Model.from_config(conf)
    patch.set_weights(base.get_weights())

    return patch


AlignedXception71 = partial(
    wrap_bone,
    aligned_xception.Xception71,
    aligned_xception.preprocess_input)


def AlignedXception71Stride16(*args, **kwargs):
    base = AlignedXception71(*args, **kwargs)
    conf = base.get_config()

    conf = patch_config(conf, ['exit_flow/block1/unit1/sepconv3_depthwise'], 'strides', (1, 1))
    conf = patch_config(conf, ['exit_flow/block1/unit1/shortcut'], 'strides', (1, 1))
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv1_depthwise'], 'dilation_rate', (2, 2))
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv2_depthwise'], 'dilation_rate', (2, 2))
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv3_depthwise'], 'dilation_rate', (2, 2))

    patch = models.Model.from_config(conf)
    patch.set_weights(base.get_weights())

    return patch


def AlignedXception71Stride8(*args, **kwargs):
    base = AlignedXception71(*args, **kwargs)
    conf = base.get_config()

    conf = patch_config(conf, ['entry_flow/block5/unit1/sepconv3_depthwise'], 'strides', (1, 1))
    conf = patch_config(conf, ['entry_flow/block5/unit1/shortcut'], 'strides', (1, 1))

    for i in range(16):
        conf = patch_config(conf, [
            'middle_flow/block1/unit{}/sepconv1_depthwise'.format(i + 1)], 'dilation_rate', (2, 2))
        conf = patch_config(conf, [
            'middle_flow/block1/unit{}/sepconv2_depthwise'.format(i + 1)], 'dilation_rate', (2, 2))
        conf = patch_config(conf, [
            'middle_flow/block1/unit{}/sepconv3_depthwise'.format(i + 1)], 'dilation_rate', (2, 2))

    conf = patch_config(conf, ['exit_flow/block1/unit1/sepconv3_depthwise'], 'strides', (1, 1))
    conf = patch_config(conf, ['exit_flow/block1/unit1/shortcut'], 'strides', (1, 1))

    # Output shape is different when using exit dilation rates like in
    # https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py#L265
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv1_depthwise'], 'dilation_rate', (2, 2))
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv2_depthwise'], 'dilation_rate', (2, 2))
    conf = patch_config(conf, ['exit_flow/block2/unit1/sepconv3_depthwise'], 'dilation_rate', (2, 2))

    patch = models.Model.from_config(conf)
    patch.set_weights(base.get_weights())

    return patch
