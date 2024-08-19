from keras import ops
from keras.src.backend import random

shape = (2, 4)
for dt in ["float16", "float32", "float64", "int16", "int32", "int64"] + [
    "int8",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]:
    # try:
    _ = ops.cast(random.uniform(shape, dtype="float32") * 3, dtype=dt)
    # except:
    #     print('-', dt)
