from functools import partial
from keras.applications import efficientnet_v2
from segme.policy.backbone.utils import wrap_bone
from segme.policy.backbone.backbone import BACKBONES

BACKBONES.register('efficientnet_v2_small')((
    partial(wrap_bone, efficientnet_v2.EfficientNetV2S, None), [
        None, 'block1b_add', 'block2d_add', 'block3d_add', 'block5i_add', 'top_activation']))

BACKBONES.register('efficientnet_v2_medium')((
    partial(wrap_bone, efficientnet_v2.EfficientNetV2M, None), [
        None, 'block1c_add', 'block2e_add', 'block3e_add', 'block5n_add', 'top_activation']))

BACKBONES.register('efficientnet_v2_large')((
    partial(wrap_bone, efficientnet_v2.EfficientNetV2L, None), [
        None, 'block1d_add', 'block2g_add', 'block3g_add', 'block5s_add', 'top_activation']))
