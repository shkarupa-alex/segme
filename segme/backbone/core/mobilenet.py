from keras.applications import mobilenet, mobilenet_v2, mobilenet_v3
from functools import partial
from ..utils import wrap_bone

MobileNet = partial(
    wrap_bone,
    mobilenet.MobileNet,
    mobilenet.preprocess_input)

MobileNetV2 = partial(
    wrap_bone,
    mobilenet_v2.MobileNetV2,
    mobilenet_v2.preprocess_input)

MobileNetV3Small = partial(
    wrap_bone,
    mobilenet_v3.MobileNetV3Small,
    mobilenet_v3.preprocess_input)

MobileNetV3Large = partial(
    wrap_bone,
    mobilenet_v3.MobileNetV3Large,
    mobilenet_v3.preprocess_input)
