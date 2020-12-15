import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from ..calibrated_focal import CalibratedFocalSigmoidCrossEntropy
from ..calibrated_focal import calibrated_focal_sigmoid_cross_entropy


def _to_logit(prob):
    logit = np.log(prob / (1.0 - prob))

    return logit


@keras_parameterized.run_all_keras_modes
class TestCalibratedFocalSigmoidCrossEntropy(keras_parameterized.TestCase):
    def test_config(self):
        bce_obj = CalibratedFocalSigmoidCrossEntropy(
            reduction=tf.keras.losses.Reduction.NONE,
            name='loss1'
        )
        self.assertEqual(bce_obj.name, 'loss1')
        self.assertEqual(bce_obj.reduction, tf.keras.losses.Reduction.NONE)

    def test_zeros(self):
        probs = tf.constant([[0.0], [0.0], [0.0]], 'float32')
        targets = tf.constant([[0], [0], [0]], 'int32')

        result = calibrated_focal_sigmoid_cross_entropy(y_true=targets, y_pred=probs)
        result = self.evaluate(result).tolist()

        self.assertAllClose(result, [0.0, 0.0, 0.0])

    def test_logits(self):
        logits = tf.constant([[_to_logit(0.97)], [_to_logit(0.45)], [_to_logit(0.03)]], 'float32')
        targets = tf.constant([[1], [1], [0]], 'int32')

        result = calibrated_focal_sigmoid_cross_entropy(y_true=targets, y_pred=logits, from_logits=True)
        result = self.evaluate(result).tolist()

        self.assertAllClose(result, [0.007614763919264078, 0.033212922513484955, 5.551136217363251e-10])

    def test_probs(self):
        probs = tf.constant([[0.97], [0.45], [0.03]], 'float32')
        targets = tf.constant([[1], [1], [0]], 'int32')

        result = calibrated_focal_sigmoid_cross_entropy(y_true=targets, y_pred=probs)
        result = self.evaluate(result).tolist()

        self.assertAllClose(result, [0.007614763919264078, 0.033212922513484955, 5.551136217363251e-10])

    def test_keras_model_compile(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(100,)),
            tf.keras.layers.Dense(5, activation='sigmoid')]
        )
        model.compile(loss='SegMe>calibrated_focal_sigmoid_cross_entropy')


if __name__ == '__main__':
    tf.test.main()
