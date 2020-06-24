import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda


def wrap_bone(model, prepr, channels, feats, init):
    input_shape = (None, None, channels)
    input_image = Input(name='image', shape=input_shape, dtype=tf.uint8)
    input_prep = Lambda(
        lambda img: prepr(tf.cast(img, tf.float32)),
        name='preprocess')(input_image)

    base_model = model(
        input_tensor=input_prep, include_top=False, weights=init)

    end_points = [ep for ep in feats if ep is not None]
    out_layers = [
        base_model.get_layer(name=name_idx).output if isinstance(name_idx, str)
        else base_model.get_layer(index=name_idx).output
        for name_idx in end_points]

    down_stack = Model(inputs=input_image, outputs=out_layers)

    return down_stack
