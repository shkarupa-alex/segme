from tensorflow.keras import applications
from functools import partial
from ..utils import wrap_bone

InceptionV3 = partial(
    wrap_bone,
    applications.InceptionV3,
    applications.inception_v3.preprocess_input)

InceptionResNetV2 = partial(
    wrap_bone,
    applications.InceptionResNetV2,
    applications.inception_resnet_v2.preprocess_input)

Xception = partial(
    wrap_bone,
    applications.Xception,
    applications.xception.preprocess_input)
