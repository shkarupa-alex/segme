import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.testing_infra import test_combinations, test_utils
from keras.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.laplace_edge import LaplaceEdgeCrossEntropy
from segme.loss.laplace_edge import laplace_edge_cross_entropy
from segme.loss.tests.test_common_loss import BINARY_LOGITS, BINARY_TARGETS, BINARY_WEIGHTS, MULTI_LOGITS, MULTI_TARGETS


@test_combinations.run_all_keras_modes
class TestLaplaceEdgeCrossEntropy(test_combinations.TestCase):
    def test_config(self):
        loss = LaplaceEdgeCrossEntropy(reduction=Reduction.NONE, name='loss1')
        self.assertEqual(loss.name, 'loss1')
        self.assertEqual(loss.reduction, Reduction.NONE)

    def test_zeros(self):
        logits = -10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'int32')

        result = laplace_edge_cross_entropy(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=1e-4)

    def test_ones(self):
        logits = 10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'int32')

        result = laplace_edge_cross_entropy(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=1e-4)

    def test_false(self):
        logits = -10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'int32')

        result = laplace_edge_cross_entropy(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=1e-4)

    def test_true(self):
        logits = 10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'int32')

        result = laplace_edge_cross_entropy(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=1e-4)

    def test_value(self):
        loss = LaplaceEdgeCrossEntropy(from_logits=True)
        result = self.evaluate(loss(BINARY_TARGETS, BINARY_LOGITS))

        # self.assertAlmostEqual(result, 3.879105925) # for zero padding and 1-channel BCE
        self.assertAlmostEqual(result, 4.122532, places=4)

    def test_weight(self):
        loss = LaplaceEdgeCrossEntropy(from_logits=True)

        result = self.evaluate(loss(BINARY_TARGETS[:, :, :2], BINARY_LOGITS[:, :, :2]))
        self.assertAlmostEqual(result, 2.8783655, places=4)

        result = self.evaluate(loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS))
        self.assertAlmostEqual(result, 3.0123634, places=4)

        result = self.evaluate(loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS * 2.))
        self.assertAlmostEqual(result, 3.0123634 * 2, places=4)

    def test_multi(self):
        loss = LaplaceEdgeCrossEntropy(from_logits=True)
        result = self.evaluate(loss(MULTI_TARGETS, MULTI_LOGITS))
        self.assertAlmostEqual(result, 3.2714694, places=4)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype('float32')
        targets = (np.random.rand(2, 224, 224, 1) > 0.5).astype('int32')

        loss = LaplaceEdgeCrossEntropy(from_logits=True)
        result0 = self.evaluate(loss(targets, probs))
        result1 = sum([self.evaluate(loss(targets[i:i + 1], probs[i:i + 1])) for i in range(2)]) / 2

        self.assertAlmostEqual(result0, result1, places=5)

    def test_model(self):
        model = models.Sequential([layers.Dense(5, activation='sigmoid')])
        model.compile(loss='SegMe>Loss>LaplaceEdgeCrossEntropy', run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.zeros((2, 16, 16, 1)), np.zeros((2, 16, 16, 1), 'int32'))
        models.Sequential.from_config(model.get_config())


if __name__ == '__main__':
    tf.test.main()
