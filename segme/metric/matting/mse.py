import tensorflow as tf
from keras.src.metrics import MeanSquaredError
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="SegMe>Metric>Matting")
class MSE(MeanSquaredError):
    def __init__(self, name="mse", dtype=None):
        """Creates a `MeanSquaredError` instance for matting task (by default downscales input by 255).

        Args:
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        dtype_true = tf.dtypes.as_dtype(y_true.dtype)
        scale_true = dtype_true.max if dtype_true.is_integer else 1.0
        y_true = tf.cast(y_true, self.dtype) / scale_true

        dtype_pred = tf.dtypes.as_dtype(y_pred.dtype)
        scale_pred = dtype_pred.max if dtype_pred.is_integer else 1.0
        y_pred = tf.cast(y_pred, self.dtype) / scale_pred

        return super().update_state(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
        )

    def result(self):
        return super().result() * 1000.0
