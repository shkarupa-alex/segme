from keras.applications import densenet
from functools import partial
from ..utils import wrap_bone

DenseNet121 = partial(
    wrap_bone,
    densenet.DenseNet121,
    densenet.preprocess_input)

DenseNet169 = partial(
    wrap_bone,
    densenet.DenseNet169,
    densenet.preprocess_input)

DenseNet201 = partial(
    wrap_bone,
    densenet.DenseNet201,
    densenet.preprocess_input)
