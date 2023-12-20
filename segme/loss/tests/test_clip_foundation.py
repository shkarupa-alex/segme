import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.src.testing_infra import test_combinations, test_utils
from keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.clip_foundation import ClipFoundationLoss
from segme.loss.clip_foundation import clip_foundation_loss
from segme.loss.tests.test_common_loss import MULTI_LOGITS, MULTI_TARGETS, MULTI_WEIGHTS


@test_combinations.run_all_keras_modes
class TestClipFoundationLoss(test_combinations.TestCase):
    def test_config(self):
        loss = ClipFoundationLoss(reduction=Reduction.NONE, name='loss1')
        self.assertEqual(loss.name, 'loss1')
        self.assertEqual(loss.reduction, Reduction.NONE)

    def test_zeros(self):
        logits = -10. * tf.one_hot(tf.zeros((3, 8, 8), 'int32'), 2, dtype='float32')
        targets = tf.concat([logits, logits], axis=-1)

        result = clip_foundation_loss(
            y_true=targets, y_pred=logits, sample_weight=None, scale=100., bias=None, temperature=(1., 1., 1.),
            weight=(1., 1., 1.))
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_ones(self):
        logits = 10. * tf.one_hot(tf.zeros((3, 8, 8), 'int32'), 2, dtype='float32')
        targets = tf.concat([logits, logits], axis=-1)

        result = clip_foundation_loss(
            y_true=targets, y_pred=logits, sample_weight=None, scale=100., bias=None, temperature=(1., 1., 1.),
            weight=(1., 1., 1.))
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_false(self):
        logits = tf.reshape(tf.range(3 * 8 * 8 * 2, dtype='float32') / (3 * 8 * 8 * 2), [3, 8, 8, 2])
        targets = tf.reverse(logits, axis=[0])
        targets = tf.concat([targets, targets], axis=-1)

        result = clip_foundation_loss(
            y_true=targets, y_pred=logits, sample_weight=None, scale=100., bias=None, temperature=(1., 1., 1.),
            weight=(1., 1., 1.))
        result = self.evaluate(result)

        self.assertAllClose(result, [2.3200176, 0.49343127, 2.7184358], atol=6e-3)

    def test_true(self):
        logits = tf.reshape(tf.range(3 * 8 * 8 * 2, dtype='float32') / (3 * 8 * 8 * 2), [3, 8, 8, 2])
        targets = tf.reverse(logits, axis=[-1])
        targets = tf.concat([targets, targets], axis=-1)

        result = clip_foundation_loss(
            y_true=targets, y_pred=logits, sample_weight=None, scale=100., bias=None, temperature=(1., 1., 1.),
            weight=(1., 1., 1.))
        result = self.evaluate(result)

        self.assertAllClose(result, [4.758, 0., 0.], atol=6e-3)

    def test_value(self):
        targets = tf.concat([
            tf.transpose(MULTI_LOGITS, [0, 3, 2, 1]),
            tf.reshape(tf.transpose(MULTI_LOGITS, [0, 2, 1, 3]), MULTI_LOGITS.shape)], axis=-1)

        loss = ClipFoundationLoss()
        result = self.evaluate(loss(targets, MULTI_LOGITS))

        self.assertAlmostEqual(result, 94.099625, places=6)

    def test_weight(self):
        targets = tf.concat([
            tf.transpose(MULTI_LOGITS, [0, 3, 2, 1]),
            tf.reshape(tf.transpose(MULTI_LOGITS, [0, 2, 1, 3]), MULTI_LOGITS.shape)], axis=-1)

        loss = ClipFoundationLoss()

        result = self.evaluate(loss(targets[:, :, :1], MULTI_LOGITS[:, :, :1]))
        self.assertAlmostEqual(result, 107.85203, places=5)

        result = self.evaluate(loss(targets, MULTI_LOGITS, MULTI_WEIGHTS))
        self.assertAlmostEqual(result, 130.912125, places=5)

        result = self.evaluate(loss(targets, MULTI_LOGITS, MULTI_WEIGHTS * 2.))
        self.assertAlmostEqual(result, 130.912125 * 2, places=5)

    # Not applicable due to batch-to-batch loss natures
    # def test_batch(self):
    #     probs = np.random.rand(2, 224, 224, 2).astype('float32')
    #     targets = np.random.rand(2, 224, 224, 2).astype('float32')
    #
    #     loss = ClipFoundationLoss()
    #     result0 = self.evaluate(loss(targets, probs))
    #     result1 = sum([self.evaluate(loss(targets[i:i + 1], probs[i:i + 1])) for i in range(2)]) / 2
    #
    #     self.assertAlmostEqual(result0, result1, places=6)

    def test_model(self):
        model = models.Sequential([layers.Dense(4, activation='linear')])
        model.compile(loss='SegMe>Loss>ClipFoundationLoss', run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.zeros((2, 8, 8, 4)), np.zeros((2, 8, 8, 8), 'float32'))
        models.Sequential.from_config(model.get_config())


if __name__ == '__main__':
    tf.test.main()
