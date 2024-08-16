from keras.src import backend, ops
from keras.src import constraints
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="SegMe>Common")
class MinConstraint(constraints.Constraint):
    def __init__(self, min_value):
        self.min_value = min_value

    def __call__(self, w):
        w = backend.convert_to_tensor(w)
        m = ops.cast(self.min_value, w.dtype)
        w = ops.minimum(w, m)

        return w

    def get_config(self):
        return {'min_value': self.min_value}
