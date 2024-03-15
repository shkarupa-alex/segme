import numpy as np
import tensorflow as tf
from tf_keras import layers, models
from tf_keras.src.testing_infra import test_combinations, test_utils
from tf_keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.normalized_focal import NormalizedFocalCrossEntropy
from segme.loss.normalized_focal import normalized_focal_cross_entropy
from segme.loss.tests.test_common_loss import BINARY_LOGITS, BINARY_TARGETS, BINARY_WEIGHTS, MULTI_LOGITS, MULTI_TARGETS


@test_combinations.run_all_keras_modes
class TestNormalizedFocalCrossEntropy(test_combinations.TestCase):
    def test_config(self):
        loss = NormalizedFocalCrossEntropy(reduction=Reduction.NONE, name='loss1')
        self.assertEqual(loss.name, 'loss1')
        self.assertEqual(loss.reduction, Reduction.NONE)

    def test_zeros(self):
        probs = tf.zeros((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'int32')

        result = normalized_focal_cross_entropy(
            y_true=targets, y_pred=probs, sample_weight=None, gamma=2, from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.69314694] * 3, atol=1e-4)

    def test_ones(self):
        probs = tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'int32')

        result = normalized_focal_cross_entropy(
            y_true=targets, y_pred=probs, sample_weight=None, gamma=2, from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.3132613] * 3, atol=1e-4)

    def test_false(self):
        probs = tf.zeros((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'int32')

        result = normalized_focal_cross_entropy(
            y_true=targets, y_pred=probs, sample_weight=None, gamma=2, from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.69314694] * 3, atol=1e-4)

    def test_true(self):
        probs = tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'int32')

        result = normalized_focal_cross_entropy(
            y_true=targets, y_pred=probs, sample_weight=None, gamma=2, from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [1.3132614] * 3, atol=1e-4)

    def test_value(self):
        loss = NormalizedFocalCrossEntropy(from_logits=True)
        result = self.evaluate(loss(BINARY_TARGETS, BINARY_LOGITS))

        self.assertAlmostEqual(result, 4.1686983)  # Not sure

    def test_weight(self):
        loss = NormalizedFocalCrossEntropy(from_logits=True)

        result = self.evaluate(loss(BINARY_TARGETS[:, :, :2], BINARY_LOGITS[:, :, :2]))
        self.assertAlmostEqual(result, 3.4507565, places=5)

        result = self.evaluate(loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS))
        self.assertAlmostEqual(result, 2.848103, places=5)

        result = self.evaluate(loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS * 2.))
        self.assertAlmostEqual(result, 2.848103 * 2., places=5)

    def test_multi(self):
        loss = NormalizedFocalCrossEntropy(from_logits=True)
        result = self.evaluate(loss(MULTI_TARGETS, MULTI_LOGITS))

        self.assertAlmostEqual(result, 8.575356, places=6)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype('float32')
        targets = (np.random.rand(2, 224, 224, 1) > 0.5).astype('int32')

        loss = NormalizedFocalCrossEntropy(from_logits=True)
        result0 = self.evaluate(loss(targets, probs))
        result1 = sum([self.evaluate(loss(targets[i:i + 1], probs[i:i + 1])) for i in range(2)]) / 2

        self.assertAlmostEqual(result0, result1, places=6)

    def test_model(self):
        model = models.Sequential([layers.Dense(1, activation='sigmoid')])
        model.compile(loss='SegMe>Loss>NormalizedFocalCrossEntropy', run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.zeros((2, 16, 16, 1)), np.zeros((2, 16, 16, 1), 'int32'))
        models.Sequential.from_config(model.get_config())


if __name__ == '__main__':
    tf.test.main()
