from functools import partial

from keras.src.applications import mobilenet_v3

from segme.policy.backbone.backbone import BACKBONES
from segme.policy.backbone.utils import wrap_bone


def hard_sigmoid(x):
    return mobilenet_v3.layers.Activation("hard_sigmoid")(x)


# TODO: wait for https://github.com/keras-team/tf-keras/issues/133
mobilenet_v3.hard_sigmoid = hard_sigmoid

BACKBONES.register("mobilenet_v3_small")(
    (
        partial(wrap_bone, mobilenet_v3.MobileNetV3Small, None),
        [
            # None, 'activation', 're_lu_1', 'activation_2', 'activation_17', 'activation_26'
            None,
            4, 19, 37, 111, 156,
        ],
    )
)

BACKBONES.register("mobilenet_v3_large")(
    (
        partial(wrap_bone, mobilenet_v3.MobileNetV3Large, None),
        [
            # None, 're_lu_1', 're_lu_5', 'activation_4', 'activation_18', 'activation_27'
            None,
            13, 31, 76, 141, 186,
        ],
    )
)
