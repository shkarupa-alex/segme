import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from ..laplace_edge import LaplaceEdgeSigmoidCrossEntropy
from ..laplace_edge import laplace_edge_sigmoid_cross_entropy


def _to_logit(prob):
    logit = np.log(prob / (1.0 - prob))

    return logit


@keras_parameterized.run_all_keras_modes
class TestLaplaceEdgeSigmoidCrossEntropy(keras_parameterized.TestCase):
    def test_config(self):
        bce_obj = LaplaceEdgeSigmoidCrossEntropy(
            reduction=tf.keras.losses.Reduction.NONE,
            name='loss1'
        )
        self.assertEqual(bce_obj.name, 'loss1')
        self.assertEqual(bce_obj.reduction, tf.keras.losses.Reduction.NONE)

    def test_zeros(self):
        probs = tf.constant([[
            [[0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0]],
        ]], 'float32')
        targets = tf.constant([[
            [[0], [0], [0]],
            [[0], [0], [0]],
            [[0], [0], [0]],
        ]], 'int32')

        result = laplace_edge_sigmoid_cross_entropy(y_true=targets, y_pred=probs)
        result = self.evaluate(result).tolist()

        self.assertAllClose(result, [[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]])

    def test_logits(self):
        logits = tf.constant([[
            [[_to_logit(0.03)], [_to_logit(0.55)], [_to_logit(0.85)]],
            [[_to_logit(0.45)], [_to_logit(0.65)], [_to_logit(0.91)]],
            [[_to_logit(0.49)], [_to_logit(0.75)], [_to_logit(0.97)]],
        ]], 'float32')
        targets = tf.constant([[
            [[0], [0], [1]],
            [[0], [1], [1]],
            [[1], [1], [1]],
        ]], 'int32')

        result = laplace_edge_sigmoid_cross_entropy(y_true=targets, y_pred=logits, from_logits=True)
        result = self.evaluate(result).tolist()

        self.assertAllClose(result, [[
            [1.00000015e-07, 2.37450123e+00, 2.74553284e-04],
            [1.66611075e+00, 1.61571205e+00, 6.03108387e-03],
            [3.18900421e-02, 1.56161431e-02, 9.63867409e-04]
        ]])

    def test_probs(self):
        probs = tf.constant([[
            [[0.03], [0.55], [0.85]],
            [[0.45], [0.65], [0.91]],
            [[0.49], [0.75], [0.97]],
        ]], 'float32')
        targets = tf.constant([[
            [[0], [0], [1]],
            [[0], [1], [1]],
            [[1], [1], [1]],
        ]], 'int32')

        result = laplace_edge_sigmoid_cross_entropy(y_true=targets, y_pred=probs)
        result = self.evaluate(result).tolist()

        self.assertAllClose(result, [[
            [1.00000015e-07, 2.37450123e+00, 2.74553284e-04],
            [1.66611075e+00, 1.61571205e+00, 6.03108387e-03],
            [3.18900421e-02, 1.56161431e-02, 9.63867409e-04]
        ]])

    def test_keras_model_compile(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(100,)),
            tf.keras.layers.Dense(5, activation='sigmoid')]
        )
        model.compile(loss='SegMe>laplace_edge_sigmoid_cross_entropy')


if __name__ == '__main__':
    tf.test.main()
