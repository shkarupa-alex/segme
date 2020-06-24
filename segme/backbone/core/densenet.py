from tensorflow.keras import applications
from functools import partial
from ..utils import wrap_bone

DenseNet121 = partial(
    wrap_bone,
    applications.DenseNet121,
    applications.densenet.preprocess_input)

DenseNet169 = partial(
    wrap_bone,
    applications.DenseNet169,
    applications.densenet.preprocess_input)

DenseNet201 = partial(
    wrap_bone,
    applications.DenseNet201,
    applications.densenet.preprocess_input)
