import numpy as np
import tensorflow as tf
from tf_keras import layers, models
from tf_keras.src.testing_infra import test_combinations, test_utils
from tf_keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.mean_absolute import MeanAbsoluteClassificationError, MeanAbsoluteRegressionError
from segme.loss.mean_absolute import mean_absolute_classification_error, mean_absolute_regression_error
from segme.loss.tests.test_common_loss import BINARY_LOGITS, BINARY_TARGETS, BINARY_WEIGHTS, MULTI_LOGITS, MULTI_TARGETS


@test_combinations.run_all_keras_modes
class TestMeanAbsoluteClassificationError(test_combinations.TestCase):
    def test_config(self):
        loss = MeanAbsoluteClassificationError(reduction=Reduction.NONE, name='loss1')
        self.assertEqual(loss.name, 'loss1')
        self.assertEqual(loss.reduction, Reduction.NONE)

    def test_zeros(self):
        logits = -10. * tf.ones((3, 64, 64, 1), 'float32')
        targets = tf.zeros((3, 64, 64, 1), 'int32')

        result = mean_absolute_classification_error(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, label_smoothing=0.)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_ones(self):
        logits = 10 * tf.ones((3, 64, 64, 1), 'float32')
        targets = tf.ones((3, 64, 64, 1), 'int32')

        result = mean_absolute_classification_error(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, label_smoothing=0.)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_false(self):
        logits = -10. * tf.ones((3, 64, 64, 1), 'float32')
        targets = tf.ones((3, 64, 64, 1), 'int32')

        result = mean_absolute_classification_error(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, label_smoothing=0.)
        result = self.evaluate(result)

        self.assertAllClose(result, [1.] * 3, atol=6e-3)

    def test_true(self):
        logits = 10. * tf.ones((3, 64, 64, 1), 'float32')
        targets = tf.zeros((3, 64, 64, 1), 'int32')

        result = mean_absolute_classification_error(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, label_smoothing=0.)
        result = self.evaluate(result)

        self.assertAllClose(result, [1.] * 3, atol=6e-3)

    def test_value(self):
        logits = BINARY_LOGITS
        targets = BINARY_TARGETS

        loss = MeanAbsoluteClassificationError(from_logits=True)
        result = self.evaluate(loss(targets, logits))

        self.assertAlmostEqual(result, 0.39642575)

    def test_weight(self):
        logits = BINARY_LOGITS
        targets = BINARY_TARGETS
        weights = BINARY_WEIGHTS

        loss = MeanAbsoluteClassificationError(from_logits=True)

        result = self.evaluate(loss(targets[:, :, :32], logits[:, :, :32]))
        self.assertAlmostEqual(result, 0.39642578, places=6)

        result = self.evaluate(loss(targets, logits, weights))
        self.assertAlmostEqual(result, 0.33698955, places=6)

        result = self.evaluate(loss(targets, logits, weights * 2.))
        self.assertAlmostEqual(result, 0.33698955 * 2, places=6)

    def test_multi(self):
        logits = MULTI_LOGITS
        targets = MULTI_TARGETS

        loss = MeanAbsoluteClassificationError(from_logits=True)
        result = self.evaluate(loss(targets, logits))

        self.assertAlmostEqual(result, 0.31354535, places=6)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype('float32')
        targets = (np.random.rand(2, 224, 224, 1) > 0.5).astype('int32')

        loss = MeanAbsoluteClassificationError(from_logits=True)
        result0 = self.evaluate(loss(targets, probs))
        result1 = sum([self.evaluate(loss(targets[i:i + 1], probs[i:i + 1])) for i in range(2)]) / 2

        self.assertAlmostEqual(result0, result1, places=6)

    def test_model(self):
        model = models.Sequential([layers.Dense(5, activation='sigmoid')])
        model.compile(loss='SegMe>Loss>MeanAbsoluteClassificationError', run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.zeros((2, 64, 64, 1)), np.zeros((2, 64, 64, 1), 'int32'))
        models.Sequential.from_config(model.get_config())


@test_combinations.run_all_keras_modes
class TestMeanAbsoluteRegressionError(test_combinations.TestCase):
    def test_config(self):
        loss = MeanAbsoluteRegressionError(
            reduction=Reduction.NONE,
            name='loss1'
        )
        self.assertEqual(loss.name, 'loss1')
        self.assertEqual(loss.reduction, Reduction.NONE)

    def test_zeros(self):
        logits = tf.zeros((3, 64, 64, 1), 'float32')
        targets = tf.zeros((3, 64, 64, 1), 'int32')

        result = mean_absolute_regression_error(y_true=targets, y_pred=logits, sample_weight=None)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_ones(self):
        logits = tf.ones((3, 64, 64, 1), 'float32')
        targets = tf.ones((3, 64, 64, 1), 'int32')

        result = mean_absolute_regression_error(y_true=targets, y_pred=logits, sample_weight=None)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_false(self):
        logits = tf.zeros((3, 64, 64, 1), 'float32')
        targets = tf.ones((3, 64, 64, 1), 'int32')

        result = mean_absolute_regression_error(y_true=targets, y_pred=logits, sample_weight=None)
        result = self.evaluate(result)

        self.assertAllClose(result, [1.] * 3, atol=6e-3)

    def test_true(self):
        logits = tf.ones((3, 64, 64, 1), 'float32')
        targets = tf.zeros((3, 64, 64, 1), 'int32')

        result = mean_absolute_regression_error(y_true=targets, y_pred=logits, sample_weight=None)
        result = self.evaluate(result)

        self.assertAllClose(result, [1.] * 3, atol=6e-3)

    def test_value(self):
        logits = tf.nn.sigmoid(BINARY_LOGITS)
        targets = tf.cast(BINARY_TARGETS, 'float32')

        loss = MeanAbsoluteRegressionError()
        result = self.evaluate(loss(targets, logits))

        self.assertAlmostEqual(result, 0.39642575)

    def test_weight(self):
        logits = tf.nn.sigmoid(BINARY_LOGITS)
        targets = tf.cast(BINARY_TARGETS, 'float32')
        weights = BINARY_WEIGHTS

        loss = MeanAbsoluteRegressionError()

        result = self.evaluate(loss(targets[:, :, :32], logits[:, :, :32]))
        self.assertAlmostEqual(result, 0.39642578, places=6)

        result = self.evaluate(loss(targets, logits, weights))
        self.assertAlmostEqual(result, 0.33698955, places=6)

        result = self.evaluate(loss(targets, logits, weights * 2.))
        self.assertAlmostEqual(result, 0.33698955 * 2, places=6)

    def test_multi(self):
        logits = tf.nn.sigmoid(MULTI_LOGITS)
        targets = tf.one_hot(tf.squeeze(MULTI_TARGETS, -1), 4, dtype='float32')

        loss = MeanAbsoluteRegressionError()
        result = self.evaluate(loss(targets, logits))

        self.assertAlmostEqual(result, 0.5163353, places=6)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype('float32')
        targets = np.random.rand(2, 224, 224, 1)

        loss = MeanAbsoluteRegressionError()
        result0 = self.evaluate(loss(targets, probs))
        result1 = sum([self.evaluate(loss(targets[i:i + 1], probs[i:i + 1])) for i in range(2)]) / 2

        self.assertAlmostEqual(result0, result1, places=6)

    def test_model(self):
        model = models.Sequential([layers.Dense(5)])
        model.compile(loss='SegMe>Loss>MeanAbsoluteRegressionError', run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.zeros((2, 64, 64, 1)), np.zeros((2, 64, 64, 5), 'float32'))
        models.Sequential.from_config(model.get_config())


if __name__ == '__main__':
    tf.test.main()
