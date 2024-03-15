from functools import partial
from tf_keras.src.applications import mobilenet_v3
from segme.policy.backbone.utils import wrap_bone
from segme.policy.backbone.backbone import BACKBONES


def hard_sigmoid(x):
    return mobilenet_v3.layers.Activation('hard_sigmoid')(x)


# TODO: wait for https://github.com/keras-team/keras/issues/15282
mobilenet_v3.hard_sigmoid = hard_sigmoid

BACKBONES.register('mobilenet_v3_small')((
    partial(wrap_bone, mobilenet_v3.MobileNetV3Small, None), [
        # None, 'multiply', 're_lu_1', 'multiply_1', 'multiply_11', 'multiply_17'
        None, 5, 20, 39, 123, 174]))

BACKBONES.register('mobilenet_v3_large')((
    partial(wrap_bone, mobilenet_v3.MobileNetV3Large, None), [
        # None, 're_lu_1', 're_lu_5', 'multiply_1', 'multiply_13', 'multiply_19'
        None, 14, 32, 78, 155, 206]))
