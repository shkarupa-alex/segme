import tensorflow as tf
import tfswin
from functools import partial
from keras import layers, models


def wrap_bone(model, channels, feats, init):
    input_shape = (None, None, channels)
    input_image = layers.Input(name='image', shape=input_shape, dtype=tf.uint8)
    input_prep = layers.Lambda(tfswin.preprocess_input, name='preprocess')(input_image)

    base_model = model(input_tensor=input_prep, include_top=False, weights=init)

    end_points = [ep for ep in feats if ep is not None]
    out_layers = [base_model.get_layer(name=name).output for name in end_points]

    down_stack = models.Model(inputs=input_image, outputs=out_layers)

    return down_stack


SwinTransformerTiny224 = partial(wrap_bone, tfswin.SwinTransformerTiny224)

SwinTransformerSmall224 = partial(wrap_bone, tfswin.SwinTransformerSmall224)

SwinTransformerBase224 = partial(wrap_bone, tfswin.SwinTransformerBase224)

SwinTransformerBase384 = partial(wrap_bone, tfswin.SwinTransformerBase384)

SwinTransformerLarge224 = partial(wrap_bone, tfswin.SwinTransformerLarge224)

SwinTransformerLarge384 = partial(wrap_bone, tfswin.SwinTransformerLarge384)
