import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.testing_infra import test_combinations, test_utils
from keras.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.consistency_enhanced import ConsistencyEnhancedLoss
from segme.loss.consistency_enhanced import consistency_enhanced_loss
from segme.loss.tests.test_common_loss import BINARY_LOGITS, BINARY_TARGETS, BINARY_WEIGHTS, MULTI_LOGITS, MULTI_TARGETS


@test_combinations.run_all_keras_modes
class TestConsistencyEnhancedLoss(test_combinations.TestCase):
    def test_config(self):
        loss = ConsistencyEnhancedLoss(
            reduction=Reduction.NONE,
            name='loss1'
        )
        self.assertEqual(loss.name, 'loss1')
        self.assertEqual(loss.reduction, Reduction.NONE)

    def test_zeros(self):
        logits = -10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'int32')

        result = consistency_enhanced_loss(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=1e-2)

    def test_ones(self):
        logits = 10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'int32')

        result = consistency_enhanced_loss(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=1e-2)

    def test_false(self):
        logits = -10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'int32')

        result = consistency_enhanced_loss(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [1.] * 3, atol=1e-2)

    def test_true(self):
        logits = 10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'int32')

        result = consistency_enhanced_loss(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [1.] * 3, atol=1e-2)

    def test_value(self):
        loss = ConsistencyEnhancedLoss(from_logits=True)
        result = self.evaluate(loss(BINARY_TARGETS, BINARY_LOGITS))

        self.assertAlmostEqual(result, 0.39642566)

    def test_weight(self):
        loss = ConsistencyEnhancedLoss(from_logits=True)

        result = self.evaluate(loss(BINARY_TARGETS[:, :, :2], BINARY_LOGITS[:, :, :2]))
        self.assertAlmostEqual(result, 0.3369894)

        result = self.evaluate(loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS))
        self.assertAlmostEqual(result, 0.33698937)

        result = self.evaluate(loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS * 2.))
        self.assertAlmostEqual(result, 0.33698946)

    def test_multi(self):
        loss = ConsistencyEnhancedLoss(from_logits=True)
        result = self.evaluate(loss(MULTI_TARGETS, MULTI_LOGITS))

        self.assertAlmostEqual(result, 0.6837686, places=6)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype('float32')
        targets = (np.random.rand(2, 224, 224, 1) > 0.5).astype('int32')

        loss = ConsistencyEnhancedLoss(from_logits=True)
        result0 = self.evaluate(loss(targets, probs))
        result1 = sum([self.evaluate(loss(targets[i:i + 1], probs[i:i + 1])) for i in range(2)]) / 2

        self.assertAlmostEqual(result0, result1)

    def test_model(self):
        model = models.Sequential([layers.Dense(5, activation='sigmoid')])
        model.compile(loss='SegMe>Loss>ConsistencyEnhancedLoss', run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.zeros((2, 16, 16, 1)), np.zeros((2, 16, 16, 1), 'int32'))
        models.Sequential.from_config(model.get_config())


if __name__ == '__main__':
    tf.test.main()
