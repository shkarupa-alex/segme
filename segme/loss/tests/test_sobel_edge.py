import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from ..sobel_edge import SobelEdgeSigmoidCrossEntropy
from ..sobel_edge import sobel_edge_sigmoid_cross_entropy


def _to_logit(prob):
    logit = np.log(prob / (1.0 - prob))

    return logit


@keras_parameterized.run_all_keras_modes
class TestSobelEdgeSigmoidCrossEntropy(keras_parameterized.TestCase):
    def test_config(self):
        bce_obj = SobelEdgeSigmoidCrossEntropy(
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

        result = sobel_edge_sigmoid_cross_entropy(y_true=targets, y_pred=probs, classes=1)
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

        result = sobel_edge_sigmoid_cross_entropy(y_true=targets, y_pred=logits, classes=1, from_logits=True)
        result = self.evaluate(result).tolist()

        self.assertAllClose(result, [[
            [0.008250176906585693, 0.0730133056640625, 0.012630999088287354],
            [0.101939857006073, 0.1887747347354889, 0.061267077922821045],
            [0.06962978839874268, 0.03966039419174194, 0.0780753493309021]
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

        result = sobel_edge_sigmoid_cross_entropy(y_true=targets, y_pred=probs, classes=1)
        result = self.evaluate(result).tolist()

        self.assertAllClose(result, [[
            [0.008250176906585693, 0.0730133056640625, 0.012630999088287354],
            [0.101939857006073, 0.1887747347354889, 0.061267077922821045],
            [0.06962978839874268, 0.03966039419174194, 0.0780753493309021]
        ]])

    def test_keras_model_compile(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(100,)),
            tf.keras.layers.Dense(5, activation='sigmoid')]
        )
        model.compile(loss='SegMe>sobel_edge_sigmoid_cross_entropy')


if __name__ == '__main__':
    tf.test.main()
