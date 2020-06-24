from tensorflow.keras import applications
from functools import partial
from ..utils import wrap_bone

MobileNet = partial(
    wrap_bone,
    applications.MobileNet,
    applications.mobilenet.preprocess_input)

MobileNetV2 = partial(
    wrap_bone,
    applications.MobileNetV2,
    applications.mobilenet_v2.preprocess_input)
