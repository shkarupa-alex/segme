from tensorflow.keras import applications
from functools import partial
from ..utils import wrap_bone

VGG16 = partial(
    wrap_bone,
    applications.VGG16,
    applications.vgg16.preprocess_input)

VGG19 = partial(
    wrap_bone,
    applications.VGG19,
    applications.vgg19.preprocess_input)
