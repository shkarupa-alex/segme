from keras.applications import efficientnet, efficientnet_v2
from functools import partial
from ..utils import wrap_bone

EfficientNetB0 = partial(
    wrap_bone,
    efficientnet.EfficientNetB0,
    efficientnet.preprocess_input)

EfficientNetB1 = partial(
    wrap_bone,
    efficientnet.EfficientNetB1,
    efficientnet.preprocess_input)

EfficientNetB2 = partial(
    wrap_bone,
    efficientnet.EfficientNetB2,
    efficientnet.preprocess_input)

EfficientNetB3 = partial(
    wrap_bone,
    efficientnet.EfficientNetB3,
    efficientnet.preprocess_input)

EfficientNetB4 = partial(
    wrap_bone,
    efficientnet.EfficientNetB4,
    efficientnet.preprocess_input)

EfficientNetB5 = partial(
    wrap_bone,
    efficientnet.EfficientNetB5,
    efficientnet.preprocess_input)

EfficientNetB6 = partial(
    wrap_bone,
    efficientnet.EfficientNetB6,
    efficientnet.preprocess_input)

EfficientNetB7 = partial(
    wrap_bone,
    efficientnet.EfficientNetB7,
    efficientnet.preprocess_input)

EfficientNetV2S = partial(
    wrap_bone,
    efficientnet_v2.EfficientNetV2S,
    None)

EfficientNetV2M = partial(
    wrap_bone,
    efficientnet_v2.EfficientNetV2M,
    None)

EfficientNetV2L = partial(
    wrap_bone,
    efficientnet_v2.EfficientNetV2L,
    None)

