from keras.metrics import MeanSquaredError
from keras.utils.generic_utils import register_keras_serializable


@register_keras_serializable(package='SegMe')
class MSE(MeanSquaredError):
    def __init__(self, divider=255., name='mse', dtype=None):
        """Creates a `MeanSquaredError` instance for matting task (by default downscales input by 255).

        Args:
            divider: A float value for input scaling.
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(name=name, dtype=dtype)
        self.divider = divider

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(
            y_true=y_true / self.divider, y_pred=y_pred / self.divider, sample_weight=sample_weight)

    def result(self):
        return super().result() * 1000.

    def get_config(self):
        config = super().get_config()
        config.update({'divider': self.divider})

        return config
