from keras.src import backend
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation

from segme import backend as back


class L2Normalize(Operation):
    def __init__(self, axis=-1, epsilon=1e-12):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon

    def compute_output_spec(self, x):
        output_dtype = backend.standardize_dtype(x.dtype)
        if "int" in output_dtype or output_dtype == "bool":
            output_dtype = backend.floatx()
        backend.KerasTensor(shape=x.shape, dtype=output_dtype)

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.l2_normalize(x, axis=self.axis, epsilon=self.epsilon)


def l2_normalize(x, axis=-1, epsilon=1e-12):
    if any_symbolic_tensors((x,)):
        return L2Normalize(axis=axis, epsilon=epsilon).symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return back.l2_normalize(x, axis=axis, epsilon=epsilon)


class Logdet(Operation):
    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.logdet(x)


def logdet(x):
    if any_symbolic_tensors((x,)):
        return Logdet().symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return back.logdet(x)


class SaturateCast(Operation):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def compute_output_spec(self, x):
        return backend.KerasTensor(shape=x.shape, dtype=self.dtype)

    def call(self, x):
        x = backend.convert_to_tensor(x)
        return back.saturate_cast(x, dtype=self.dtype)


def saturate_cast(x, dtype):
    if any_symbolic_tensors((x,)):
        return SaturateCast(dtype=dtype).symbolic_call(x)
    x = backend.convert_to_tensor(x)
    return back.saturate_cast(x, dtype=dtype)
