import tensorflow as tf
from keras.src import models
from keras.src.backend.tensorflow.core import convert_to_tensor
from tensorflow.python.framework import convert_to_constants

# TODO: bias_add, linear+bias, conv+bias


def op_type(x):
    x = convert_to_tensor(x)
    if isinstance(x, (tf.__internal__.EagerTensor, tf.Variable)):
        return None

    if not hasattr(x, "op") or not hasattr(x.op, "type"):
        return None

    return x.op.type


def model_inference_fn(model, jit_compile):
    if not isinstance(model, models.Functional):
        raise ValueError(
            f"Expecting model to be an instance of `keras.models.Functional`. "
            f"Got: {type(model)}"
        )

    if isinstance(model.inputs, dict):
        input_spec = {
            k: tf.TensorSpec(v.shape, v.dtype) for k, v in model.inputs.items()
        }
    else:
        input_spec = [tf.TensorSpec(i.shape, i.dtype) for i in model.inputs]
        if isinstance(model.inputs, tuple):
            input_spec = tuple(input_spec)

    model_fn = tf.function(
        lambda *args: model(*args, training=False),
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    model_fn = model_fn.get_concrete_function(input_spec)
    model_fn = convert_to_constants.convert_variables_to_constants_v2(model_fn)

    return model_fn
