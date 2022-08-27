import tfgcvit
from functools import partial
from segme.policy.backbone.utils import wrap_bone
from segme.policy.backbone.backbone import BACKBONES

GCVIT_ENDPOINTS = [
    None, 'patch_embed/conv_down/norm1', 'levels/0/downsample/norm1', 'levels/1/downsample/norm1',
    'levels/2/downsample/norm1', 'norm']

BACKBONES.register('gcvit_nano')((
    partial(wrap_bone, tfgcvit.GCViTNano, tfgcvit.preprocess_input), GCVIT_ENDPOINTS))

BACKBONES.register('gcvit_micro')((
    partial(wrap_bone, tfgcvit.GCViTMicro, tfgcvit.preprocess_input), GCVIT_ENDPOINTS))

BACKBONES.register('gcvit_tiny')((
    partial(wrap_bone, tfgcvit.GCViTTiny, tfgcvit.preprocess_input), GCVIT_ENDPOINTS))

BACKBONES.register('gcvit_small')((
    partial(wrap_bone, tfgcvit.GCViTSmall, tfgcvit.preprocess_input), GCVIT_ENDPOINTS))

BACKBONES.register('gcvit_base')((
    partial(wrap_bone, tfgcvit.GCViTBase, tfgcvit.preprocess_input), GCVIT_ENDPOINTS))
