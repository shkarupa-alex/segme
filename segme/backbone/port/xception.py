from functools import partial
from ..utils import wrap_bone
from . import aligned_xception

AlignedXception41 = partial(
    wrap_bone,
    aligned_xception.Xception41,
    aligned_xception.preprocess_input)

AlignedXception65 = partial(
    wrap_bone,
    aligned_xception.Xception65,
    aligned_xception.preprocess_input)

AlignedXception71 = partial(
    wrap_bone,
    aligned_xception.Xception71,
    aligned_xception.preprocess_input)
