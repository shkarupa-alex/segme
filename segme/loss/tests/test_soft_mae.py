import numpy as np
import tensorflow as tf
from tf_keras import layers, models
from tf_keras.src.testing_infra import test_combinations, test_utils
from tf_keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.soft_mae import SoftMeanAbsoluteError, soft_mean_absolute_error
from segme.loss.tests.test_common_loss import BINARY_LOGITS, BINARY_TARGETS, BINARY_WEIGHTS, MULTI_LOGITS, MULTI_TARGETS


@test_combinations.run_all_keras_modes
class TestSoftMeanAbsoluteError(test_combinations.TestCase):
    def test_config(self):
        loss = SoftMeanAbsoluteError(
            reduction=Reduction.NONE,
            name='loss1'
        )
        self.assertEqual(loss.name, 'loss1')
        self.assertEqual(loss.reduction, Reduction.NONE)

    def test_zeros(self):
        logits = tf.zeros((3, 64, 64, 1), 'float32')
        targets = tf.zeros((3, 64, 64, 1), 'int32')

        result = soft_mean_absolute_error(y_true=targets, y_pred=logits, beta=1., sample_weight=None)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_ones(self):
        logits = tf.ones((3, 64, 64, 1), 'float32')
        targets = tf.ones((3, 64, 64, 1), 'int32')

        result = soft_mean_absolute_error(y_true=targets, y_pred=logits, beta=1., sample_weight=None)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_false(self):
        logits = tf.zeros((3, 64, 64, 1), 'float32')
        targets = tf.ones((3, 64, 64, 1), 'int32')

        result = soft_mean_absolute_error(y_true=targets, y_pred=logits, beta=1., sample_weight=None)
        result = self.evaluate(result)

        self.assertAllClose(result, [.5] * 3, atol=6e-3)

    def test_true(self):
        logits = tf.ones((3, 64, 64, 1), 'float32')
        targets = tf.zeros((3, 64, 64, 1), 'int32')

        result = soft_mean_absolute_error(y_true=targets, y_pred=logits, beta=1., sample_weight=None)
        result = self.evaluate(result)

        self.assertAllClose(result, [.5] * 3, atol=6e-3)

    def test_value(self):
        logits = np.arange(-10, 11.)[:, None].astype('float32') / 2.
        targets = np.zeros_like(logits)
        expected = np.array([
            4., 3.5, 3., 2.5, 2., 1.5, 1., 0.5625, 0.25, 0.0625, 0.,
            0.0625, 0.25, 0.5625, 1., 1.5, 2., 2.5, 3., 3.5, 4.]).astype('float32')

        loss = SoftMeanAbsoluteError(beta=2., reduction='none')
        result = self.evaluate(loss(targets, logits))

        self.assertAllEqual(result, expected)

    def test_weight(self):
        logits = tf.nn.sigmoid(BINARY_LOGITS)
        targets = tf.cast(BINARY_TARGETS, 'float32')
        weights = BINARY_WEIGHTS

        loss = SoftMeanAbsoluteError()

        result = self.evaluate(loss(targets[:, :, :32], logits[:, :, :32]))
        self.assertAlmostEqual(result, 0.16333841, places=6)

        result = self.evaluate(loss(targets, logits, weights))
        self.assertAlmostEqual(result, 0.12487066, places=6)

        result = self.evaluate(loss(targets, logits, weights * 2.))
        self.assertAlmostEqual(result, 0.12487066 * 2, places=6)

    def test_multi(self):
        logits = tf.nn.sigmoid(MULTI_LOGITS)
        targets = tf.one_hot(tf.squeeze(MULTI_TARGETS, -1), 4, dtype='float32')

        loss = SoftMeanAbsoluteError()
        result = self.evaluate(loss(targets, logits))

        self.assertAlmostEqual(result, 0.2127596, places=6)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype('float32')
        targets = np.random.rand(2, 224, 224, 1)

        loss = SoftMeanAbsoluteError()
        result0 = self.evaluate(loss(targets, probs))
        result1 = sum([self.evaluate(loss(targets[i:i + 1], probs[i:i + 1])) for i in range(2)]) / 2

        self.assertAlmostEqual(result0, result1, places=6)

    def test_model(self):
        model = models.Sequential([layers.Dense(5)])
        model.compile(loss='SegMe>Loss>SoftMeanAbsoluteError', run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.zeros((2, 64, 64, 5)), np.zeros((2, 64, 64, 5), 'float32'))
        models.Sequential.from_config(model.get_config())


if __name__ == '__main__':
    tf.test.main()
