import numpy as np
import tensorflow as tf
from tensorflow.python.framework import sparse_tensor


def get_shape(inputs, axis=None, dtype="int32", name=None):
    with tf.name_scope(name or "get_shape"):
        if not tf.is_tensor(inputs):
            inputs = tf.convert_to_tensor(inputs)

        axis = list(axis or range(inputs.shape.rank))
        axis = [inputs.shape.rank * int(a < 0) + a for a in axis]

        dtype = tf.as_dtype(dtype)
        if not dtype.is_numpy_compatible or not (
            dtype.is_floating or dtype.is_integer
        ):
            raise ValueError("Requested shape dtype is not supported.")

        static_shape = []
        for i, s in enumerate(inputs.shape.as_list()):
            if i not in axis:
                continue
            if s is not None:
                s = np.array(s, dtype.as_numpy_dtype).item()
            static_shape.append(s)

        fully_defined = None not in static_shape
        if fully_defined:
            return static_shape, fully_defined

        if dtype in {tf.int32, tf.int64}:
            dynamic_shape = tf.shape(inputs, out_type=dtype)
        else:
            dynamic_shape = tf.shape(inputs, out_type="int32")
            dynamic_shape = tf.cast(dynamic_shape, dtype)

        if 1 == static_shape.count(None):
            mixed_shape = list(static_shape)
            axis_idx = mixed_shape.index(None)
            axis_val = axis[axis_idx]
            mixed_shape[axis_idx] = dynamic_shape[axis_val]

            return mixed_shape, fully_defined

        dynamic_shape = tf.unstack(dynamic_shape)
        dynamic_shape = [dynamic_shape[a] for a in axis]
        dynamic_shape = [
            d if s is None else s for s, d in zip(static_shape, dynamic_shape)
        ]

        return dynamic_shape, fully_defined
