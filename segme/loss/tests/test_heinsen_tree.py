import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.src.testing_infra import test_combinations, test_utils
from keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.heinsen_tree import HeinsenTreeLoss
from segme.loss.heinsen_tree import heinsen_tree_loss


@test_combinations.run_all_keras_modes
class TestHeinsenTreeLoss(test_combinations.TestCase):
    def test_config(self):
        loss = HeinsenTreeLoss([], reduction=Reduction.NONE, name='loss1')
        self.assertEqual(loss.name, 'loss1')
        self.assertEqual(loss.reduction, Reduction.NONE)

    # def test_zeros(self):
    #     logits = -10. * tf.ones((3, 6), 'float32')
    #     targets = tf.zeros((3, 1), 'int32')
    #
    #     result = heinsen_tree_loss(
    #         y_true=targets, y_pred=logits, sample_weight=None, tree_paths=TREE_PATHS, crossentropy='categorical',
    #         label_smoothing=0., from_logits=True)
    #     result = self.evaluate(result)
    #
    #     self.assertAllClose(result, [0.693147] * 3, atol=1e-4)
    #
    # def test_ones(self):
    #     logits = 10. * tf.ones((3, 6), 'float32')
    #     targets = tf.ones((3, 1), 'int32')
    #
    #     result = heinsen_tree_loss(
    #         y_true=targets, y_pred=logits, sample_weight=None, tree_paths=TREE_PATHS, crossentropy='categorical',
    #         label_smoothing=0., from_logits=True)
    #     result = self.evaluate(result)
    #
    #     self.assertAllClose(result, [0.693147] * 3, atol=1e-4)
    #
    # def test_false(self):
    #     logits = -10. * tf.ones((3, 6), 'float32')
    #     targets = tf.ones((3, 1), 'int32')
    #
    #     result = heinsen_tree_loss(
    #         y_true=targets, y_pred=logits, sample_weight=None, tree_paths=TREE_PATHS, crossentropy='categorical',
    #         label_smoothing=0., from_logits=True)
    #     result = self.evaluate(result)
    #
    #     self.assertAllClose(result, [0.693147] * 3, atol=1e-4)
    #
    # def test_true(self):
    #     logits = 10. * tf.ones((3, 6), 'float32')
    #     targets = tf.zeros((3, 1), 'int32')
    #
    #     result = heinsen_tree_loss(
    #         y_true=targets, y_pred=logits, sample_weight=None, tree_paths=TREE_PATHS, crossentropy='categorical',
    #         label_smoothing=0., from_logits=True)
    #     result = self.evaluate(result)
    #
    #     self.assertAllClose(result, [0.693147] * 3, atol=1e-4)

    def test_value(self):
        loss = HeinsenTreeLoss(TREE_PATHS, from_logits=True)
        result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS))

        self.assertAlmostEqual(result, 0.82880473, places=6)  # 0.8316239714622498 with non-segment mean

    # def test_value_smooth(self):
    #     loss = HeinsenTreeLoss(TREE_PATHS, label_smoothing=1e-5, from_logits=True)
    #     result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS))
    #
    #     self.assertAlmostEqual(result, 0.82947063, places=5)
    #
    #     loss = HeinsenTreeLoss(TREE_PATHS, label_smoothing=0.1, from_logits=True)
    #     result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS))
    #
    #     self.assertAlmostEqual(result, 7.4887037, places=5)
    #
    # def test_value_binary(self):
    #     loss = HeinsenTreeLoss(TREE_PATHS, crossentropy='binary', from_logits=True)
    #     result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS))
    #
    #     self.assertAlmostEqual(result, 0.2545368, places=6)
    #
    # def test_value_2d(self):
    #     loss = HeinsenTreeLoss(TREE_PATHS, from_logits=True)
    #     targets = tf.reshape(TREE_TARGETS, [2, 2, 1])
    #     logits = tf.reshape(TREE_LOGITS, [2, 2, 6])
    #     result = self.evaluate(loss(targets, logits))
    #
    #     self.assertAlmostEqual(result, 0.82880473, places=6)
    #
    # def test_weight(self):
    #     weights = tf.concat([tf.ones((2, 1)), tf.zeros((2, 1))], axis=0)
    #
    #     loss = HeinsenTreeLoss(TREE_PATHS, from_logits=True)
    #
    #     result = self.evaluate(loss(TREE_TARGETS[:2], TREE_LOGITS[:2]))
    #     self.assertAlmostEqual(result, 0.3177374)
    #
    #     result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS, weights))
    #     self.assertAlmostEqual(result, 0.1588687)
    #
    #     result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS, weights * 2.))
    #     self.assertAlmostEqual(result, 0.1588687 * 2., places=6)
    #
    # def test_weight_2d(self):
    #     targets = tf.reshape(TREE_TARGETS, [2, 2, 1])
    #     logits = tf.reshape(TREE_LOGITS, [2, 2, 6])
    #     weights = tf.concat([tf.ones((1, 2)), tf.zeros((1, 2))], axis=0)
    #
    #     loss = HeinsenTreeLoss(TREE_PATHS, from_logits=True)
    #
    #     result = self.evaluate(loss(targets[:1], logits[:1]))
    #     self.assertAlmostEqual(result, 0.3177374)
    #
    #     result = self.evaluate(loss(targets, logits, weights))
    #     self.assertAlmostEqual(result, 0.1588687)
    #
    #     result = self.evaluate(loss(targets, logits, weights * 2.))
    #     self.assertAlmostEqual(result, 0.1588687 * 2., places=6)
    #
    # def test_multi_probs(self):
    #     probs = 1 / (1 + np.exp(-TREE_LOGITS))
    #     probs = tf.constant(probs)
    #     probs._keras_logits = TREE_LOGITS
    #
    #     loss = HeinsenTreeLoss(TREE_PATHS)
    #     result = self.evaluate(loss(TREE_TARGETS, probs))
    #
    #     self.assertAlmostEqual(result, 0.82880473, places=6)
    #
    # def test_batch(self):
    #     targets = tf.reshape(TREE_TARGETS, [2, 2, 1])
    #     logits = tf.reshape(TREE_LOGITS, [2, 2, 6])
    #
    #     loss = HeinsenTreeLoss(TREE_PATHS, from_logits=True)
    #     result0 = self.evaluate(loss(targets, logits))
    #     result1 = sum([self.evaluate(loss(targets[i:i + 1], logits[i:i + 1])) for i in range(2)]) / 2
    #
    #     self.assertAlmostEqual(result0, result1)
    #
    # def test_model(self):
    #     model = models.Sequential([layers.Dense(6, activation='sigmoid')])
    #     model.compile(loss=HeinsenTreeLoss(TREE_PATHS), run_eagerly=test_utils.should_run_eagerly())
    #     model.fit(np.zeros((2, 6)), np.zeros((2, 1), 'int32'))
    #     models.Sequential.from_config(model.get_config())


TREE_PATHS = [[0, 3, 4], [0], [1], [0, 2], [0, 3], [0, 3, 5]]
TREE_LOGITS = tf.constant([
    [-1.134725, -1.853591, -1.793681, -0.253602, 0.649923, -0.321642],
    [-0.545508, 0.388591, 0.335042, -0.300561, -0.918901, -0.152296],
    [-0.837255, 1.03309, -0.190284, 0.837352, 0.257713, 1.214522],
    [-0.813726, 1.356368, -1.323276, -0.316505, 0.125477, -1.459085]], 'float32')
TREE_TARGETS = tf.constant([[4], [1], [5], [2]], 'int32')

if __name__ == '__main__':
    tf.test.main()
