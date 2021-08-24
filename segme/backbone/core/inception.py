from keras.applications import inception_v3, inception_resnet_v2, xception
from functools import partial
from ..utils import wrap_bone

InceptionV3 = partial(
    wrap_bone,
    inception_v3.InceptionV3,
    inception_v3.preprocess_input)

InceptionResNetV2 = partial(
    wrap_bone,
    inception_resnet_v2.InceptionResNetV2,
    inception_resnet_v2.preprocess_input)

Xception = partial(
    wrap_bone,
    xception.Xception,
    xception.preprocess_input)
