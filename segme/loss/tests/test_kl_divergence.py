import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.src.testing_infra import test_combinations, test_utils
from keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.kl_divergence import KLDivergenceLoss
from segme.loss.kl_divergence import kl_divergence_loss
from segme.loss.tests.test_common_loss import MULTI_LOGITS, MULTI_TARGETS, MULTI_WEIGHTS


@test_combinations.run_all_keras_modes
class TestKLDivergenceLoss(test_combinations.TestCase):
    def test_config(self):
        loss = KLDivergenceLoss(reduction=Reduction.NONE, name='loss1')
        self.assertEqual(loss.name, 'loss1')
        self.assertEqual(loss.reduction, Reduction.NONE)

    def test_zeros(self):
        logits = -10. * tf.one_hot(tf.zeros((3, 8, 8), 'int32'), 2, dtype='float32')
        targets = logits

        result = kl_divergence_loss(y_true=targets, y_pred=logits, sample_weight=None, temperature=1.)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_ones(self):
        logits = 10. * tf.one_hot(tf.zeros((3, 8, 8), 'int32'), 2, dtype='float32')
        targets = logits

        result = kl_divergence_loss(y_true=targets, y_pred=logits, sample_weight=None, temperature=1.)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_false(self):
        logits = -10. * tf.one_hot(tf.zeros((3, 8, 8), 'int32'), 2, dtype='float32')
        targets = tf.reverse(logits, axis=[-1])

        result = kl_divergence_loss(y_true=targets, y_pred=logits, sample_weight=None, temperature=1.)
        result = self.evaluate(result)

        self.assertAllClose(result, [9.999] * 3, atol=6e-3)

    def test_true(self):
        logits = 10. * tf.one_hot(tf.zeros((3, 8, 8), 'int32'), 2, dtype='float32')
        targets = tf.reverse(logits, axis=[-1])

        result = kl_divergence_loss(y_true=targets, y_pred=logits, sample_weight=None, temperature=1.)
        result = self.evaluate(result)

        self.assertAllClose(result, [9.999] * 3, atol=6e-3)

    def test_value(self):
        targets = tf.one_hot(tf.squeeze(MULTI_TARGETS, axis=-1), MULTI_LOGITS.shape[-1])
        loss = KLDivergenceLoss()
        result = self.evaluate(loss(targets, MULTI_LOGITS))

        self.assertAlmostEqual(result, 3.9633336, places=6)

    def test_weight(self):
        targets = tf.one_hot(tf.squeeze(MULTI_TARGETS, axis=-1), MULTI_LOGITS.shape[-1])

        loss = KLDivergenceLoss()

        result = self.evaluate(loss(targets[:, :, :1], MULTI_LOGITS[:, :, :1]))
        self.assertAlmostEqual(result, 3.4156132, places=5)

        result = self.evaluate(loss(targets, MULTI_LOGITS, MULTI_WEIGHTS))
        self.assertAlmostEqual(result, 3.4156132, places=5)

        result = self.evaluate(loss(targets, MULTI_LOGITS, MULTI_WEIGHTS * 2.))
        self.assertAlmostEqual(result, 3.4156132 * 2, places=5)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 2).astype('float32')
        targets = np.random.rand(2, 224, 224, 2).astype('float32')

        loss = KLDivergenceLoss()
        result0 = self.evaluate(loss(targets, probs))
        result1 = sum([self.evaluate(loss(targets[i:i + 1], probs[i:i + 1])) for i in range(2)]) / 2

        self.assertAlmostEqual(result0, result1, places=6)

    def test_model(self):
        model = models.Sequential([layers.Dense(10, activation='linear')])
        model.compile(loss='SegMe>Loss>KLDivergenceLoss', run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.zeros((2, 8, 8, 1)), np.zeros((2, 8, 8, 10), 'float32'))
        models.Sequential.from_config(model.get_config())


if __name__ == '__main__':
    tf.test.main()
