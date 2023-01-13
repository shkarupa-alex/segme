import tfswin
from functools import partial
from keras import models
from segme.policy.backbone.utils import wrap_bone
from segme.policy.backbone.backbone import BACKBONES

SWIN_ENDPOINTS = [None, None, 'layers.0', 'layers.1', 'layers.2', 'norm']


def wrap_bone_norm(model, prepr, init, channels, end_points, name):
    base_model = wrap_bone(model, prepr, init, channels, end_points, name)

    output_feats = [tfswin.norm.LayerNorm(name=f'features_{i}_norm')(out)
                    for i, out in enumerate(base_model.outputs[:-1])]
    output_feats.append(base_model.outputs[-1])
    down_stack = models.Model(inputs=base_model.inputs, outputs=output_feats, name=name)

    return down_stack


BACKBONES.register('swin_tiny_224')((
    partial(wrap_bone_norm, tfswin.SwinTransformerTiny224, tfswin.preprocess_input), SWIN_ENDPOINTS))

BACKBONES.register('swin_small_224')((
    partial(wrap_bone_norm, tfswin.SwinTransformerSmall224, tfswin.preprocess_input), SWIN_ENDPOINTS))

BACKBONES.register('swin_base_224')((
    partial(wrap_bone_norm, tfswin.SwinTransformerBase224, tfswin.preprocess_input), SWIN_ENDPOINTS))

BACKBONES.register('swin_base_384')((
    partial(wrap_bone_norm, tfswin.SwinTransformerBase384, tfswin.preprocess_input), SWIN_ENDPOINTS))

BACKBONES.register('swin_large_224')((
    partial(wrap_bone_norm, tfswin.SwinTransformerLarge224, tfswin.preprocess_input), SWIN_ENDPOINTS))

BACKBONES.register('swin_large_384')((
    partial(wrap_bone_norm, tfswin.SwinTransformerLarge384, tfswin.preprocess_input), SWIN_ENDPOINTS))

BACKBONES.register('swin_v2_tiny_256')((
    partial(wrap_bone_norm, tfswin.SwinTransformerV2Tiny256, tfswin.preprocess_input), SWIN_ENDPOINTS))

BACKBONES.register('swin_v2_small_256')((
    partial(wrap_bone_norm, tfswin.SwinTransformerV2Small256, tfswin.preprocess_input), SWIN_ENDPOINTS))

BACKBONES.register('swin_v2_base_256')((
    partial(wrap_bone_norm, tfswin.SwinTransformerV2Base256, tfswin.preprocess_input), SWIN_ENDPOINTS))

BACKBONES.register('swin_v2_base_384')((
    partial(wrap_bone_norm, tfswin.SwinTransformerV2Base384, tfswin.preprocess_input), SWIN_ENDPOINTS))

BACKBONES.register('swin_v2_large_256')((
    partial(wrap_bone_norm, tfswin.SwinTransformerV2Large256, tfswin.preprocess_input), SWIN_ENDPOINTS))

BACKBONES.register('swin_v2_large_384')((
    partial(wrap_bone_norm, tfswin.SwinTransformerV2Large384, tfswin.preprocess_input), SWIN_ENDPOINTS))
