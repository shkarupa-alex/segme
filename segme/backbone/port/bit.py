from functools import partial
from keras import models
from ..utils import patch_config, wrap_bone
from . import big_transfer

BiT_S_R50x1 = partial(
    wrap_bone,
    big_transfer.BiT_S_R50x1,
    big_transfer.preprocess_input)

BiT_S_R50x3 = partial(
    wrap_bone,
    big_transfer.BiT_S_R50x3,
    big_transfer.preprocess_input)

BiT_S_R101x1 = partial(
    wrap_bone,
    big_transfer.BiT_S_R101x1,
    big_transfer.preprocess_input)

BiT_S_R101x3 = partial(
    wrap_bone,
    big_transfer.BiT_S_R101x3,
    big_transfer.preprocess_input)

BiT_S_R152x4 = partial(
    wrap_bone,
    big_transfer.BiT_S_R152x4,
    big_transfer.preprocess_input)

BiT_M_R50x1 = partial(
    wrap_bone,
    big_transfer.BiT_M_R50x1,
    big_transfer.preprocess_input)


def BiT_M_R50x1Stride8(*args, **kwargs):
    base = BiT_M_R50x1(*args, **kwargs)
    conf = base.get_config()

    conf = patch_config(conf, ['block3', 'unit01'], 'stride', 1)
    conf = patch_config(conf, ['block3', 'unit02'], 'dilation', 2)
    conf = patch_config(conf, ['block3', 'unit03'], 'dilation', 2)
    conf = patch_config(conf, ['block3', 'unit04'], 'dilation', 2)
    conf = patch_config(conf, ['block3', 'unit05'], 'dilation', 2)
    conf = patch_config(conf, ['block3', 'unit06'], 'dilation', 2)

    conf = patch_config(conf, ['block4', 'unit01'], 'stride', 1)
    conf = patch_config(conf, ['block4', 'unit01'], 'dilation', 2)
    conf = patch_config(conf, ['block4', 'unit02'], 'dilation', 4)
    conf = patch_config(conf, ['block4', 'unit03'], 'dilation', 4)

    patch = models.Model.from_config(conf)
    patch.set_weights(base.get_weights())

    return patch


BiT_M_R50x3 = partial(
    wrap_bone,
    big_transfer.BiT_M_R50x3,
    big_transfer.preprocess_input)

BiT_M_R101x1 = partial(
    wrap_bone,
    big_transfer.BiT_M_R101x1,
    big_transfer.preprocess_input)

BiT_M_R101x3 = partial(
    wrap_bone,
    big_transfer.BiT_M_R101x3,
    big_transfer.preprocess_input)

BiT_M_R152x4 = partial(
    wrap_bone,
    big_transfer.BiT_M_R152x4,
    big_transfer.preprocess_input)
