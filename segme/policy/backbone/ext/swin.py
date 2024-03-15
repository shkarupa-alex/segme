import tfswin
from functools import partial
from segme.policy.backbone.utils import wrap_bone
from segme.policy.backbone.backbone import BACKBONES

SWIN_ENDPOINTS = [None, None, 'layers.0', 'layers.1', 'layers.2', 'layers.3']

BACKBONES.register('swin_tiny_224')((
    partial(wrap_bone, tfswin.SwinTransformerTiny224, None), SWIN_ENDPOINTS))

BACKBONES.register('swin_small_224')((
    partial(wrap_bone, tfswin.SwinTransformerSmall224, None), SWIN_ENDPOINTS))

BACKBONES.register('swin_base_224')((
    partial(wrap_bone, tfswin.SwinTransformerBase224, None), SWIN_ENDPOINTS))

BACKBONES.register('swin_base_384')((
    partial(wrap_bone, tfswin.SwinTransformerBase384, None), SWIN_ENDPOINTS))

BACKBONES.register('swin_large_224')((
    partial(wrap_bone, tfswin.SwinTransformerLarge224, None), SWIN_ENDPOINTS))

BACKBONES.register('swin_large_384')((
    partial(wrap_bone, tfswin.SwinTransformerLarge384, None), SWIN_ENDPOINTS))

BACKBONES.register('swin_v2_tiny_256')((
    partial(wrap_bone, tfswin.SwinTransformerV2Tiny256, None), SWIN_ENDPOINTS))

BACKBONES.register('swin_v2_small_256')((
    partial(wrap_bone, tfswin.SwinTransformerV2Small256, None), SWIN_ENDPOINTS))

BACKBONES.register('swin_v2_base_256')((
    partial(wrap_bone, tfswin.SwinTransformerV2Base256, None), SWIN_ENDPOINTS))

BACKBONES.register('swin_v2_base_384')((
    partial(wrap_bone, tfswin.SwinTransformerV2Base384, None), SWIN_ENDPOINTS))

BACKBONES.register('swin_v2_large_256')((
    partial(wrap_bone, tfswin.SwinTransformerV2Large256, None), SWIN_ENDPOINTS))

BACKBONES.register('swin_v2_large_384')((
    partial(wrap_bone, tfswin.SwinTransformerV2Large384, None), SWIN_ENDPOINTS))
