import cv2
import tensorflow as tf
from keras.metrics import BinaryAccuracy, SparseCategoricalAccuracy
from keras.saving.object_registration import register_keras_serializable


@register_keras_serializable(package='SegMe>Metric>Boundary')
class BinaryBoundaryAccuracy(BinaryAccuracy):
    def __init__(self, radius=1, threshold=0.5, name='binary_boundary_accuracy', dtype=None):
        """Creates an `Accuracy` metric instance estimated only in `radius` pixels from boundary.

        Args:
            radius: (Optional) int radius of boundary
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(name, threshold=threshold, dtype=dtype)
        self.radius = radius

    def update_state(self, y_true, y_pred, sample_weight=None):
        sample_weight = boundary_weight(y_true, self.radius, sample_weight)

        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({'radius': self.radius})

        return config


@register_keras_serializable(package='SegMe>Metric>Boundary')
class SparseCategoricalBoundaryAccuracy(SparseCategoricalAccuracy):
    def __init__(self, radius=1, name='sparse_categorical_boundary_accuracy', dtype=None):
        """Creates a `SparseCategoricalAccuracy` metric instance estimated only in `radius` pixels from boundary.

        Args:
            radius: (Optional) int radius of boundary
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(name, dtype=dtype)
        self.radius = radius

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_1h = tf.one_hot(tf.squeeze(y_true, -1), y_pred.shape[-1], dtype='int32')
        sample_weight = boundary_weight(y_true_1h, self.radius, sample_weight)

        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({'radius': self.radius})

        return config


def boundary_weight(y_true, radius, sample_weight):
    if 4 != len(y_true.shape):
        raise ValueError(f'Labels must have rank 4.')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))[..., None]
    kernel = tf.convert_to_tensor(kernel, 'int32')
    kernel = tf.tile(kernel, (1, 1, y_true.shape[-1]))

    background = tf.cast(y_true != 0, 'int32')

    def _cond(i, e, d):
        return i < radius

    def _body(i, e, d):
        return i + 1, \
               tf.nn.erosion2d(e, kernel, [1] * 4, 'SAME', 'NHWC', [1] * 4), \
               tf.nn.dilation2d(d, kernel, [1] * 4, 'SAME', 'NHWC', [1] * 4)

    _, eroded, dilated = tf.while_loop(_cond, _body, [0, background, background])

    weight = tf.cast(eroded + dilated == 1, 'float32')
    weight = tf.reduce_max(weight, axis=-1, keepdims=True)

    batch, height, width, _ = tf.unstack(tf.shape(y_true))
    frame = tf.zeros((batch, height - radius * 2, width - radius * 2, 1), 'float32')
    frame = tf.pad(frame, [[0, 0], [radius, radius], [radius, radius], [0, 0]], constant_values=1.)
    weight = tf.maximum(weight, frame)

    if sample_weight is None:
        sample_weight = weight
    else:
        sample_weight *= weight

    return sample_weight
