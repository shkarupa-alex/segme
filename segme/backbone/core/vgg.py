from keras.applications import vgg16, vgg19
from functools import partial
from ..utils import wrap_bone

VGG16 = partial(
    wrap_bone,
    vgg16.VGG16,
    vgg16.preprocess_input)

VGG19 = partial(
    wrap_bone,
    vgg19.VGG19,
    vgg19.preprocess_input)
