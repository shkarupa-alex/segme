from tensorflow.python.keras.applications import efficientnet
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
