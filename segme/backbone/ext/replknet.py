import tfreplknet
from functools import partial
from ..utils import wrap_bone

RepLKNetB224In1k = partial(wrap_bone, tfreplknet.RepLKNetB224In1k, tfreplknet.preprocess_input_bl)

RepLKNetB224In21k = partial(wrap_bone, tfreplknet.RepLKNetB224In21k, tfreplknet.preprocess_input_bl)

RepLKNetB384In1k = partial(wrap_bone, tfreplknet.RepLKNetB384In1k, tfreplknet.preprocess_input_bl)

RepLKNetL384In1k = partial(wrap_bone, tfreplknet.RepLKNetL384In1k, tfreplknet.preprocess_input_bl)

RepLKNetL384In21k = partial(wrap_bone, tfreplknet.RepLKNetL384In21k, tfreplknet.preprocess_input_bl)

RepLKNetXL320In1k = partial(wrap_bone, tfreplknet.RepLKNetXL320In1k, tfreplknet.preprocess_input_xl)

RepLKNetXL320In21k = partial(wrap_bone, tfreplknet.RepLKNetXL320In21k, tfreplknet.preprocess_input_xl)
