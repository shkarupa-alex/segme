import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from ..laplace_edge import LaplaceSigmoidEdgeHold
from ..laplace_edge import laplace_sigmoid_edge_hold


def _to_logit(prob):
    logit = np.log(prob / (1.0 - prob))

    return logit


def _log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))

    return numerator / denominator


@keras_parameterized.run_all_keras_modes
class TestLaplaceSigmoidEdgeHold(keras_parameterized.TestCase):
    def test_config(self):
        bce_obj = LaplaceSigmoidEdgeHold(
            reduction=tf.keras.losses.Reduction.NONE,
            name='laplace_sigmoid_edge_hold1'
        )
        self.assertEqual(bce_obj.name, 'laplace_sigmoid_edge_hold1')
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
        ]], tf.float32)

        fl = laplace_sigmoid_edge_hold(
            y_true=target_tensor,
            y_pred=prediction_tensor,
            from_logits=True
        )
        fl = self.evaluate(fl).tolist()

        self.assertAllClose(fl, [[
            [1.00000015e-07, 2.37450123e+00, 2.74553284e-04],
            [1.66611075e+00, 1.61571205e+00, 6.03108387e-03],
            [3.18900421e-02, 1.56161431e-02, 9.63867409e-04]
        ]])

    def test_keras_model_compile(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(100,)),
            tf.keras.layers.Dense(5, activation='sigmoid')]
        )
        model.compile(loss='SegMe>laplace_sigmoid_edge_hold')

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
        ]], tf.float32)

        fl = laplace_sigmoid_edge_hold(y_true=target_tensor, y_pred=prediction_tensor)
        fl = self.evaluate(fl).tolist()

        self.assertAllClose(fl, [[
            [3.63653040e+00, 3.09840989e+00, 1.57127809e-03],
            [2.90886593e+00, 2.70888805e+00, 2.24102922e-02],
            [4.88529168e-03, 3.18960622e-02, 1.72067818e-03]
        ]], atol=1e-8)


if __name__ == '__main__':
    tf.test.main()
