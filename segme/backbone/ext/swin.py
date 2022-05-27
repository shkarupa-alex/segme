import tfswin
import tensorflow as tf
from functools import partial
from keras import backend, layers, models


def wrap_bone(model, channels, feats, init, trainable):
    if 3 != channels:
        raise ValueError('Unsupported channel size')

    input_image = layers.Input(name='image', shape=(None, None, channels), dtype='uint8')
    input_prep = layers.Lambda(tfswin.preprocess_input, name='preprocess')(input_image)

    base_model = model(input_tensor=input_prep, include_top=False, weights=init)

    end_points = [ep for ep in feats if ep is not None]
    out_layers = [base_model.get_layer(name=name).output for name in end_points]

    down_stack = models.Model(inputs=input_image, outputs=out_layers)
    down_stack.trainable = trainable

    out_layers = [tfswin.norm.LayerNorm(name=f'feature_norm_{i}')(out) for i, out in enumerate(down_stack.outputs)]
    down_stack = models.Model(inputs=down_stack.inputs, outputs=out_layers)

    return down_stack


SwinTransformerTiny224 = partial(wrap_bone, tfswin.SwinTransformerTiny224)

SwinTransformerSmall224 = partial(wrap_bone, tfswin.SwinTransformerSmall224)

SwinTransformerBase224 = partial(wrap_bone, tfswin.SwinTransformerBase224)

SwinTransformerBase384 = partial(wrap_bone, tfswin.SwinTransformerBase384)

SwinTransformerLarge224 = partial(wrap_bone, tfswin.SwinTransformerLarge224)

SwinTransformerLarge384 = partial(wrap_bone, tfswin.SwinTransformerLarge384)

SwinTransformerV2Tiny256 = partial(wrap_bone, tfswin.SwinTransformerV2Tiny256)

SwinTransformerV2Small256 = partial(wrap_bone, tfswin.SwinTransformerV2Small256)

SwinTransformerV2Base256 = partial(wrap_bone, tfswin.SwinTransformerV2Base256)

SwinTransformerV2Base384 = partial(wrap_bone, tfswin.SwinTransformerV2Base384)

SwinTransformerV2Large256 = partial(wrap_bone, tfswin.SwinTransformerV2Large256)

SwinTransformerV2Large384 = partial(wrap_bone, tfswin.SwinTransformerV2Large384)
