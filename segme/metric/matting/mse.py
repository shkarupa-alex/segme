from keras.src.metrics import MeanSquaredError
from keras.src.saving import register_keras_serializable

from segme.ops import convert_image_dtype


@register_keras_serializable(package="SegMe>Metric>Matting")
class MSE(MeanSquaredError):
    def __init__(self, name="mse", dtype=None):
        """Creates a `MeanSquaredError` instance for matting task (by default
        downscales input by 255).

        Args:
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = convert_image_dtype(y_true, self.dtype)
        y_pred = convert_image_dtype(y_pred, self.dtype)

        return super().update_state(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
        )

    def result(self):
        return super().result() * 1000.0
