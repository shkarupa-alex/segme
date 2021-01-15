from functools import partial
from ..utils import wrap_bone
from . import big_transfer

BiT_S_R50x1 = partial(
    wrap_bone,
    big_transfer.BiT_S_R50x1,
    big_transfer.preprocess_input)

BiT_S_R50x3 = partial(
    wrap_bone,
    big_transfer.BiT_S_R50x3,
    big_transfer.preprocess_input)

BiT_S_R101x1 = partial(
    wrap_bone,
    big_transfer.BiT_S_R101x1,
    big_transfer.preprocess_input)

BiT_S_R101x3 = partial(
    wrap_bone,
    big_transfer.BiT_S_R101x3,
    big_transfer.preprocess_input)

BiT_S_R152x4 = partial(
    wrap_bone,
    big_transfer.BiT_S_R152x4,
    big_transfer.preprocess_input)

BiT_M_R50x1 = partial(
    wrap_bone,
    big_transfer.BiT_M_R50x1,
    big_transfer.preprocess_input)

BiT_M_R50x3 = partial(
    wrap_bone,
    big_transfer.BiT_M_R50x3,
    big_transfer.preprocess_input)

BiT_M_R101x1 = partial(
    wrap_bone,
    big_transfer.BiT_M_R101x1,
    big_transfer.preprocess_input)

BiT_M_R101x3 = partial(
    wrap_bone,
    big_transfer.BiT_M_R101x3,
    big_transfer.preprocess_input)

BiT_M_R152x4 = partial(
    wrap_bone,
    big_transfer.BiT_M_R152x4,
    big_transfer.preprocess_input)
