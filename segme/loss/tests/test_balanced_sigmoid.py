import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from ..balanced_sigmoid import BalancedSigmoidCrossEntropy
from ..balanced_sigmoid import balanced_sigmoid_cross_entropy


def _to_logit(prob):
    logit = np.log(prob / (1.0 - prob))

    return logit


def _log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))

    return numerator / denominator


@keras_parameterized.run_all_keras_modes
class TestBalancedSigmoidCrossEntropy(keras_parameterized.TestCase):
    def test_config(self):
        bce_obj = BalancedSigmoidCrossEntropy(
            reduction=tf.keras.losses.Reduction.NONE,
            name='balanced_sigmoid_cross_entropy1'
        )
        self.assertEqual(bce_obj.name, 'balanced_sigmoid_cross_entropy1')
        self.assertEqual(bce_obj.reduction, tf.keras.losses.Reduction.NONE)

    def test_logits(self):
        prediction_tensor = tf.constant([
            [_to_logit(0.97)],
            [_to_logit(0.45)],
            [_to_logit(0.03)],
        ], tf.float32)
        target_tensor = tf.constant([[1], [1], [0]], tf.float32)

        fl = balanced_sigmoid_cross_entropy(
            y_true=target_tensor,
            y_pred=prediction_tensor,
            from_logits=True
        )
        fl = self.evaluate(fl).tolist()

        self.assertAllClose(fl, [0.01015307, 0.26616925, 0.02030611])

    def test_keras_model_compile(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(100,)),
            tf.keras.layers.Dense(5, activation='sigmoid')]
        )
        model.compile(loss='SegMe>balanced_sigmoid_cross_entropy')

    def test_sigmoid(self):
        prediction_tensor = tf.constant([[0.97], [0.45], [0.03]], tf.float32)
        prediction_tensor = tf.nn.sigmoid(prediction_tensor)
        target_tensor = tf.constant([[1], [1], [0]], tf.float32)

        fl = balanced_sigmoid_cross_entropy(y_true=target_tensor, y_pred=prediction_tensor)
        fl = self.evaluate(fl).tolist()

        self.assertAllClose(fl, [2.0559804e-07, 3.3212923e-02, 5.5511362e-10], atol=1e-8)


if __name__ == '__main__':
    tf.test.main()
