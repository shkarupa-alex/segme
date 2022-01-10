import numpy as np
import tensorflow as tf
from keras import keras_parameterized, layers, models
from keras.utils.losses_utils import ReductionV2 as Reduction
from ..normalized_focal import NormalizedFocalSigmoidCrossEntropy
from ..normalized_focal import normalized_focal_sigmoid_cross_entropy


def _to_logit(prob):
    logit = np.log(prob / (1.0 - prob))

    return logit


@keras_parameterized.run_all_keras_modes
class TestNormalizedFocalSigmoidCrossEntropy(keras_parameterized.TestCase):
    def test_config(self):
        bce_obj = NormalizedFocalSigmoidCrossEntropy(
            reduction=Reduction.NONE,
            name='loss1'
        )
        self.assertEqual(bce_obj.name, 'loss1')
        self.assertEqual(bce_obj.reduction, Reduction.NONE)

    def test_zeros(self):
        probs = tf.zeros((2, 4, 1), 'float32')
        targets = tf.zeros((2, 4, 1), 'int32')

        result = normalized_focal_sigmoid_cross_entropy(
            y_true=targets, y_pred=probs, alpha=0.25, gamma=2, from_logits=False)
        result = self.evaluate(result).tolist()

        self.assertAllClose(result, np.zeros((2, 4)))

    def test_weight_4d(self):
        logits = tf.constant([
            [[[0.4250706654827763], [7.219920928747051], [7.14131948950217], [2.5576064452206024]],
             [[1.342442193620409], [0.20020616879804165], [3.977300484664198], [6.280817910206608]],
             [[0.3206719246447576], [3.0176225602425912], [2.902292891065069], [3.369106587128292]],
             [[2.6576544216404563], [6.863726154333165], [4.581314280496405], [7.433728759092233]]],
            [[[8.13888654097292], [8.311411218599392], [0.8372454481780323], [2.859455217953778]],
             [[2.0984725413538854], [4.619268334888168], [8.708732477440673], [1.9102341271004541]],
             [[3.4914178176388266], [4.551627675234152], [7.709902261544302], [3.3982255596983277]],
             [[0.9182162683255968], [3.0387004793287886], [2.1883984916630697], [1.3921544038795197]]]], 'float32')
        targets = tf.constant([
            [[[0], [0], [1], [0]], [[1], [0], [1], [1]], [[0], [1], [0], [1]], [[0], [1], [1], [1]]],
            [[[0], [1], [1], [0]], [[1], [0], [0], [1]], [[0], [1], [1], [0]], [[1], [1], [1], [1]]]], 'int32')
        weights = tf.concat([tf.ones((2, 4, 2, 1)), tf.zeros((2, 4, 2, 1))], axis=2)

        loss = NormalizedFocalSigmoidCrossEntropy(from_logits=True, reduction=Reduction.SUM)

        result = self.evaluate(loss(targets, logits)).item()
        self.assertAlmostEqual(result, 100.46464538574219, places=3)

        result = self.evaluate(loss(targets, logits, weights)).item()
        self.assertAlmostEqual(result, 58.79029083251953, places=4)

    def test_logits(self):
        logits = tf.constant([
            [[_to_logit(0.97)], [_to_logit(0.45)]], [[_to_logit(0.55)], [_to_logit(0.03)]]], 'float32')
        targets = tf.constant([[[1], [1]], [[1], [0]]], 'int32')

        result = normalized_focal_sigmoid_cross_entropy(
            y_true=targets, y_pred=logits, alpha=0.25, gamma=2, from_logits=True)
        result = self.evaluate(result).tolist()

        self.assertAllClose(result, [
            [4.517669731285423e-05, 0.39806923270225525], [0.2975955605506897, 0.00020216232223901898]])

    def test_probs(self):
        probs = tf.constant([[[0.97], [0.45]], [[0.55], [0.03]]], 'float32')
        targets = tf.constant([[[1], [1]], [[1], [0]]], 'int32')

        result = normalized_focal_sigmoid_cross_entropy(
            y_true=targets, y_pred=probs, alpha=0.25, gamma=2, from_logits=False)
        result = self.evaluate(result).tolist()

        self.assertAllClose(result, [
            [4.517669731285423e-05, 0.39806923270225525], [0.2975955605506897, 0.00020216232223901898]])

    def test_keras_model_compile(self):
        model = models.Sequential([
            layers.Input(shape=(100,)),
            layers.Dense(5, activation='sigmoid')]
        )
        model.compile(loss='SegMe>normalized_focal_sigmoid_cross_entropy')


if __name__ == '__main__':
    tf.test.main()
