import tensorflow as tf
import tfreplknet
from functools import partial
from keras import layers, models
from ..utils import wrap_bone

RepLKNet31B224K1 = partial(wrap_bone, tfreplknet.RepLKNet31B224K1, tfreplknet.preprocess_input)

RepLKNet31B224K21 = partial(wrap_bone, tfreplknet.RepLKNet31B224K21, tfreplknet.preprocess_input)

RepLKNet31B384K1 = partial(wrap_bone, tfreplknet.RepLKNet31B384K1, tfreplknet.preprocess_input)

RepLKNet31L384K1 = partial(wrap_bone, tfreplknet.RepLKNet31L384K1, tfreplknet.preprocess_input)

RepLKNet31L384K21 = partial(wrap_bone, tfreplknet.RepLKNet31L384K21, tfreplknet.preprocess_input)
