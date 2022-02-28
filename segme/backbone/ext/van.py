import tensorflow as tf
import tfvan
from functools import partial
from keras import layers, models
from ..utils import wrap_bone

VanTiny = partial(wrap_bone, tfvan.VanTiny, tfvan.preprocess_input)

VanSmall = partial(wrap_bone, tfvan.VanSmall, tfvan.preprocess_input)

VanBase = partial(wrap_bone, tfvan.VanBase, tfvan.preprocess_input)

VanLarge = partial(wrap_bone, tfvan.VanLarge, tfvan.preprocess_input)
