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

    def test_zeros_categorical(self):
        logits = -100. * tf.ones((3, 18), 'float32')
        targets = tf.zeros((3, 1), 'int32')

        result = heinsen_tree_loss(
            y_true=targets, y_pred=logits, sample_weight=None, tree_paths=TREE_PATHS, force_binary=False,
            label_smoothing=0., level_weighting='mean', from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [2.8903718] * 3, atol=1e-4)

    def test_zeros_binary(self):
        logits = -100. * tf.ones((3, 18), 'float32')
        targets = tf.zeros((3, 1), 'int32')

        result = heinsen_tree_loss(
            y_true=targets, y_pred=logits, sample_weight=None, tree_paths=TREE_PATHS, force_binary=True,
            label_smoothing=0., level_weighting='mean', from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [100.] * 3, atol=1e-4)

    def test_ones_categorical(self):
        logits = 100. * tf.ones((3, 18), 'float32')
        targets = tf.ones((3, 1), 'int32')

        result = heinsen_tree_loss(
            y_true=targets, y_pred=logits, sample_weight=None, tree_paths=TREE_PATHS, force_binary=False,
            label_smoothing=0., level_weighting='mean', from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [1.0986123] * 3, atol=1e-4)

    def test_ones_binary(self):
        logits = 100. * tf.ones((3, 18), 'float32')
        targets = tf.ones((3, 1), 'int32')

        result = heinsen_tree_loss(
            y_true=targets, y_pred=logits, sample_weight=None, tree_paths=TREE_PATHS, force_binary=True,
            label_smoothing=0., level_weighting='mean', from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [200.] * 3, atol=1e-4)

    def test_false_categorical(self):
        result = heinsen_tree_loss(
            y_true=TREE_TARGETS, y_pred=-TRUE_LOGITS, sample_weight=None, tree_paths=TREE_PATHS,
            force_binary=False, label_smoothing=0., level_weighting='mean', from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [201.3418, 200.69315, 201.03972, 201.00633], atol=1e-4)

    def test_false_binary(self):
        result = heinsen_tree_loss(
            y_true=TREE_TARGETS, y_pred=-TRUE_LOGITS, sample_weight=None, tree_paths=TREE_PATHS, force_binary=True,
            label_smoothing=0., level_weighting='mean', from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [533.3334, 300., 400., 450.], atol=1e-4)

    def test_true_categorical(self):
        result = heinsen_tree_loss(
            y_true=TREE_TARGETS, y_pred=TRUE_LOGITS, sample_weight=None, tree_paths=TREE_PATHS,
            force_binary=False, label_smoothing=0., level_weighting='mean', from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 4, atol=1e-4)

    def test_true_binary(self):
        result = heinsen_tree_loss(
            y_true=TREE_TARGETS, y_pred=TRUE_LOGITS, sample_weight=None, tree_paths=TREE_PATHS, force_binary=True,
            label_smoothing=0., level_weighting='mean', from_logits=True)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 4, atol=1e-4)

    def test_value_categorical(self):
        loss = HeinsenTreeLoss(TREE_PATHS, force_binary=False, from_logits=True)
        result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS))

        self.assertAlmostEqual(result, 4.5775123, places=6)

    def test_value_binary(self):
        loss = HeinsenTreeLoss(TREE_PATHS, force_binary=True, from_logits=True)
        result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS))

        self.assertAlmostEqual(result, 7.641514, places=6)

    def test_value_categorical_smooth(self):
        loss = HeinsenTreeLoss(TREE_PATHS, force_binary=False, label_smoothing=1e-5, from_logits=True)
        result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS))

        self.assertAlmostEqual(result, 4.577481, places=5)

        loss = HeinsenTreeLoss(TREE_PATHS, force_binary=False, label_smoothing=0.1, from_logits=True)
        result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS))

        self.assertAlmostEqual(result, 4.2565556, places=5)

    def test_value_binary_smooth(self):
        loss = HeinsenTreeLoss(TREE_PATHS, force_binary=True, label_smoothing=1e-5, from_logits=True)
        result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS))

        self.assertAlmostEqual(result, 7.641534, places=5)

        loss = HeinsenTreeLoss(TREE_PATHS, force_binary=True, label_smoothing=0.1, from_logits=True)
        result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS))

        self.assertAlmostEqual(result, 7.8363056, places=5)

    def test_value_level_linear(self):
        loss = HeinsenTreeLoss(TREE_PATHS, level_weighting='linear', from_logits=True)
        result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS))

        self.assertAlmostEqual(result, 5.158496, places=6)

    def test_value_log(self):
        loss = HeinsenTreeLoss(TREE_PATHS, level_weighting='log', from_logits=True)
        result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS))

        self.assertAlmostEqual(result, 4.924034, places=6)

    def test_value_pow(self):
        loss = HeinsenTreeLoss(TREE_PATHS, level_weighting='pow', from_logits=True)
        result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS))

        self.assertAlmostEqual(result, 4.8634114, places=6)

    def test_value_cumsum(self):
        loss = HeinsenTreeLoss(TREE_PATHS, level_weighting='cumsum', from_logits=True)
        result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS))

        self.assertAlmostEqual(result, 5.54681, places=6)

    def test_value_2d(self):
        loss = HeinsenTreeLoss(TREE_PATHS, from_logits=True)
        targets = tf.reshape(TREE_TARGETS, [2, 2, 1])
        logits = tf.reshape(TREE_LOGITS, [2, 2, 18])
        result = self.evaluate(loss(targets, logits))

        self.assertAlmostEqual(result, 4.5775123, places=6)

    def test_weight(self):
        weights = tf.concat([tf.ones((2, 1)), tf.zeros((2, 1))], axis=0)

        loss = HeinsenTreeLoss(TREE_PATHS, from_logits=True)

        result = self.evaluate(loss(TREE_TARGETS[:2], TREE_LOGITS[:2]))
        self.assertAlmostEqual(result, 3.7674043)

        result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS, weights))
        self.assertAlmostEqual(result, 1.8837022)

        result = self.evaluate(loss(TREE_TARGETS, TREE_LOGITS, weights * 2.))
        self.assertAlmostEqual(result, 1.8837022 * 2., places=6)

    def test_weight_2d(self):
        targets = tf.reshape(TREE_TARGETS, [2, 2, 1])
        logits = tf.reshape(TREE_LOGITS, [2, 2, 18])
        weights = tf.concat([tf.ones((1, 2)), tf.zeros((1, 2))], axis=0)

        loss = HeinsenTreeLoss(TREE_PATHS, from_logits=True)

        result = self.evaluate(loss(targets[:1], logits[:1]))
        self.assertAlmostEqual(result, 3.7674043)

        result = self.evaluate(loss(targets, logits, weights))
        self.assertAlmostEqual(result, 1.8837022)

        result = self.evaluate(loss(targets, logits, weights * 2.))
        self.assertAlmostEqual(result, 1.8837022 * 2., places=6)

    def test_multi_probs(self):
        probs = 1 / (1 + np.exp(-TREE_LOGITS))
        probs = tf.constant(probs)
        probs._keras_logits = TREE_LOGITS

        loss = HeinsenTreeLoss(TREE_PATHS)
        result = self.evaluate(loss(TREE_TARGETS, probs))

        self.assertAlmostEqual(result, 4.5775123, places=6)

    def test_batch(self):
        targets = tf.reshape(TREE_TARGETS, [2, 2, 1])
        logits = tf.reshape(TREE_LOGITS, [2, 2, 18])

        loss = HeinsenTreeLoss(TREE_PATHS, from_logits=True)
        result0 = self.evaluate(loss(targets, logits))
        result1 = sum([self.evaluate(loss(targets[i:i + 1], logits[i:i + 1])) for i in range(2)]) / 2

        self.assertAlmostEqual(result0, result1, places=6)

    def test_model(self):
        model = models.Sequential([layers.Dense(18, activation='sigmoid')])
        model.compile(loss=HeinsenTreeLoss(TREE_PATHS), run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.zeros((2, 18)), np.zeros((2, 1), 'int32'))
        models.Sequential.from_config(model.get_config())


#     ╭───────┼──────╮
#     0       1      2
#   ╭─┼─╮       ╭────┴────╮
#   3 4 5       6         7
# ╭─┼─╮       ╭─┴─╮    ╭──┼──╮
# 8 9 10      11 12    13 14 15
#                          ╭─┴─╮
#                          16 17
TREE_PATHS = [
    [0], [1], [2], [0, 3], [0, 4], [0, 5], [2, 6], [2, 7], [0, 3, 8], [0, 3, 9], [0, 3, 10], [2, 6, 11], [2, 6, 12],
    [2, 7, 13], [2, 7, 14], [2, 7, 15], [2, 7, 15, 16], [2, 7, 15, 17]]
TREE_LOGITS = tf.constant([
    [-2.5, 8.2, -2.9, 6.8, -0.8, -3.5, -7.3, 4.2, 5.6, -3.3, -9., -1.2, 0.1, 6.5, -6.6, -5.5, -5.1, -4.7],
    [-7.4, -1.7, 1.8, -1.6, -0.9, -0.5, -2.4, 8.1, -8.7, 1.1, -7., 3.4, 4.2, 2.9, -3., -7.3, -2.7, 3.5],
    [-3.8, 1.3, 5.4, 3.5, 3., 8., -1.8, 8.4, -4.6, -0.8, -7.2, -0.7, 4.2, 8.8, -5., 4.4, 1.2, 2.7],
    [-8., -1.8, 3.2, -4., 0.1, -7.5, -4.7, -4.2, -4.5, -4.4, -8.5, -5.4, 2.7, -3.7, -4.8, 0.2, 6.1, -8.6]], 'float32')
TREE_TARGETS = tf.constant([[8], [1], [6], [17]], 'int32')
TRUE_LOGITS = 100. * tf.constant([
    [1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 9, 9],
    [-1, 1, -1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [-1, -1, 1, -1, -1, -1, 1, -1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    [-1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1]], 'float32')

if __name__ == '__main__':
    tf.test.main()
