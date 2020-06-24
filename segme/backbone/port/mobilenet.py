from functools import partial
from ..utils import wrap_bone
from . import mobilenet_v3

MobileNetV3Small = partial(
    wrap_bone,
    mobilenet_v3.MobileNetV3Small,
    mobilenet_v3.preprocess_input)

MobileNetV3Large = partial(
    wrap_bone,
    mobilenet_v3.MobileNetV3Large,
    mobilenet_v3.preprocess_input)
