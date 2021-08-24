from keras.applications import mobilenet, mobilenet_v2
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
