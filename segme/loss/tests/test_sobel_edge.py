import numpy as np
import tensorflow as tf
from tf_keras import layers, models
from tf_keras.src.testing_infra import test_combinations, test_utils
from tf_keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.sobel_edge import SobelEdgeLoss
from segme.loss.sobel_edge import sobel_edge_loss
from segme.loss.tests.test_common_loss import BINARY_LOGITS, BINARY_TARGETS, BINARY_WEIGHTS, MULTI_LOGITS, MULTI_TARGETS


@test_combinations.run_all_keras_modes
class TestSobelEdgeLoss(test_combinations.TestCase):
    def test_config(self):
        loss = SobelEdgeLoss(reduction=Reduction.NONE, name='loss1')
        self.assertEqual(loss.name, 'loss1')
        self.assertEqual(loss.reduction, Reduction.NONE)

    def test_zeros(self):
        logits = -10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'int32')

        result = sobel_edge_loss(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=1e-4)

    def test_ones(self):
        logits = 10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'int32')

        result = sobel_edge_loss(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=1e-4)

    def test_false(self):
        logits = -10. * tf.ones((3, 6, 6, 1), 'float32')
        targets = tf.ones((3, 6, 6, 1), 'int32')

        result = sobel_edge_loss(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=1e-4)

    def test_true(self):
        logits = 10. * tf.ones((3, 6, 6, 1), 'float32')
        targets = tf.zeros((3, 6, 6, 1), 'int32')

        result = sobel_edge_loss(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=1e-4)

    def test_value(self):
        loss = SobelEdgeLoss(from_logits=True)
        result = self.evaluate(loss(BINARY_TARGETS, BINARY_LOGITS))
        self.assertAlmostEqual(result, 0.08220904)  # 0.071708525 with zero padding

    def test_weight(self):
        loss = SobelEdgeLoss(from_logits=True)

        result = self.evaluate(loss(BINARY_TARGETS[:, :, :2], BINARY_LOGITS[:, :, :2]))
        self.assertAlmostEqual(result, 0.08495622)

        result = self.evaluate(loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS))
        self.assertAlmostEqual(result, 0.0928311)

        result = self.evaluate(loss(BINARY_TARGETS, BINARY_LOGITS, BINARY_WEIGHTS * 2.))
        self.assertAlmostEqual(result, 0.0928311 * 2.)

    def test_multi(self):
        loss = SobelEdgeLoss(from_logits=True)
        result = self.evaluate(loss(MULTI_TARGETS, MULTI_LOGITS))
        self.assertAlmostEqual(result, 0.14044482)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype('float32')
        targets = (np.random.rand(2, 224, 224, 1) > 0.5).astype('int32')

        loss = SobelEdgeLoss(from_logits=True)
        result0 = self.evaluate(loss(targets, probs))
        result1 = sum([self.evaluate(loss(targets[i:i + 1], probs[i:i + 1])) for i in range(2)]) / 2

        self.assertAlmostEqual(result0, result1)

    def test_model(self):
        model = models.Sequential([layers.Dense(5, activation='sigmoid')])
        model.compile(loss='SegMe>Loss>SobelEdgeLoss', run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.zeros((2, 16, 16, 1)), np.zeros((2, 16, 16, 1), 'int32'))
        models.Sequential.from_config(model.get_config())


if __name__ == '__main__':
    tf.test.main()
