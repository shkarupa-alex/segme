from keras.applications import resnet, resnet_v2
from functools import partial
from ..utils import wrap_bone

ResNet50 = partial(
    wrap_bone,
    resnet.ResNet50,
    resnet.preprocess_input)

ResNet101 = partial(
    wrap_bone,
    resnet.ResNet101,
    resnet.preprocess_input)

ResNet152 = partial(
    wrap_bone,
    resnet.ResNet152,
    resnet.preprocess_input)

ResNet50V2 = partial(
    wrap_bone,
    resnet_v2.ResNet50V2,
    resnet_v2.preprocess_input)

ResNet101V2 = partial(
    wrap_bone,
    resnet_v2.ResNet101V2,
    resnet_v2.preprocess_input)

ResNet152V2 = partial(
    wrap_bone,
    resnet_v2.ResNet152V2,
    resnet_v2.preprocess_input)
