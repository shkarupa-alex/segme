import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from ..ppa_binary import PixelPositionAwareBinaryLoss
from ..ppa_binary import pixel_position_aware_binary_loss


def _to_logit(prob):
    logit = np.log(prob / (1.0 - prob))

    return logit


def _log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))

    return numerator / denominator


@keras_parameterized.run_all_keras_modes
class TestPixelPositionAwareBinaryLoss(keras_parameterized.TestCase):
    def test_config(self):
        bce_obj = PixelPositionAwareBinaryLoss(
            reduction=tf.keras.losses.Reduction.NONE,
            name='pixel_position_aware_binary_loss1'
        )
        self.assertEqual(bce_obj.name, 'pixel_position_aware_binary_loss1')
        self.assertEqual(bce_obj.reduction, tf.keras.losses.Reduction.NONE)

    def test_logits(self):
        prediction_tensor = tf.constant([[
            [[_to_logit(0.03)], [_to_logit(0.55)], [_to_logit(0.85)]],
            [[_to_logit(0.45)], [_to_logit(0.65)], [_to_logit(0.91)]],
            [[_to_logit(0.49)], [_to_logit(0.75)], [_to_logit(0.97)]],
        ]], tf.float32)
        target_tensor = tf.constant([[
            [[0], [0], [1]],
            [[0], [1], [1]],
            [[1], [1], [1]],
        ]], tf.int32)

        fl = pixel_position_aware_binary_loss(
            y_true=target_tensor,
            y_pred=prediction_tensor,
            from_logits=True,
            ksize=2
        )
        fl = self.evaluate(fl).tolist()

        self.assertAllClose(fl, [0.49940788745880127])

    def test_keras_model_compile(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(100,)),
            tf.keras.layers.Dense(5, activation='sigmoid')]
        )
        model.compile(loss='SegMe>pixel_position_aware_binary_loss')

    def test_sigmoid(self):
        prediction_tensor = tf.constant([[
            [[0.03], [0.55], [0.85]],
            [[0.45], [0.65], [0.91]],
            [[0.49], [0.75], [0.97]],
        ]], tf.float32)
        prediction_tensor = tf.nn.sigmoid(prediction_tensor)
        target_tensor = tf.constant([[
            [[0], [0], [1]],
            [[0], [1], [1]],
            [[1], [1], [1]],
        ]], tf.int32)

        fl = pixel_position_aware_binary_loss(y_true=target_tensor, y_pred=prediction_tensor, ksize=2)
        fl = self.evaluate(fl).tolist()

        self.assertAllClose(fl, [0.6891778707504272], atol=1e-8)


if __name__ == '__main__':
    tf.test.main()
