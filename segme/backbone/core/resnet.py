from tensorflow.keras import applications
from functools import partial
from ..utils import wrap_bone

ResNet50 = partial(
    wrap_bone,
    applications.ResNet50,
    applications.resnet.preprocess_input)

ResNet101 = partial(
    wrap_bone,
    applications.ResNet101,
    applications.resnet.preprocess_input)

ResNet152 = partial(
    wrap_bone,
    applications.ResNet152,
    applications.resnet.preprocess_input)

ResNet50V2 = partial(
    wrap_bone,
    applications.ResNet50V2,
    applications.resnet_v2.preprocess_input)

ResNet101V2 = partial(
    wrap_bone,
    applications.ResNet101V2,
    applications.resnet_v2.preprocess_input)

ResNet152V2 = partial(
    wrap_bone,
    applications.ResNet152V2,
    applications.resnet_v2.preprocess_input)
