import numpy as np
import tensorflow as tf
from keras.src.testing_infra import test_combinations
from segme.metric.boundary.bacc import BinaryBoundaryAccuracy, SparseCategoricalBoundaryAccuracy


@test_combinations.run_all_keras_modes
class TestBinaryBoundaryAccuracy(test_combinations.TestCase):
    DIAG = np.array([
        [0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]]).astype('int32')

    def test_config(self):
        metric = BinaryBoundaryAccuracy(radius=1, name='metric1')
        self.assertEqual(metric.radius, 1)
        self.assertEqual(metric.name, 'metric1')

    def test_zeros(self):
        targets = np.zeros((2, 32, 32, 1), 'int32')
        probs = np.zeros((2, 32, 32, 1), 'float32')

        metric = BinaryBoundaryAccuracy()
        metric.update_state(targets, probs)
        result = self.evaluate(metric.result())
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_ones(self):
        targets = np.ones((2, 32, 32, 1), 'int32')
        probs = np.ones((2, 32, 32, 1), 'float32')

        metric = BinaryBoundaryAccuracy()
        metric.update_state(targets, probs)
        result = self.evaluate(metric.result())
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_false(self):
        pred = (self.DIAG + np.eye(9, dtype='int32')).T

        metric = BinaryBoundaryAccuracy()
        metric.update_state(self.DIAG[None, ..., None], pred[None, ..., None])
        result = self.evaluate(metric.result())
        self.assertAlmostEqual(result, 0.0, places=7)

    def test_true(self):
        metric = BinaryBoundaryAccuracy()
        metric.update_state(self.DIAG[None, ..., None], self.DIAG[None, ..., None])
        result = self.evaluate(metric.result())
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_value(self):
        pred = self.DIAG.copy()
        pred[4, 4] = 1

        metric = BinaryBoundaryAccuracy()
        metric.update_state(self.DIAG[None, ..., None], pred[None, ..., None])
        result = self.evaluate(metric.result())

        self.assertAlmostEqual(result, 0.9777778, places=6)  # 0.94117647 without frame

    def test_batch(self):
        pred = self.DIAG.copy()
        pred[4, 4] = 1

        metric = BinaryBoundaryAccuracy()
        metric.update_state(self.DIAG[None, ..., None], pred[None, ..., None])
        metric.update_state(self.DIAG.T[None, ..., None], pred.T[None, ..., None])
        res0 = self.evaluate(metric.result())

        metric.reset_states()
        metric.update_state(np.stack([self.DIAG, self.DIAG.T])[..., None], np.stack([pred, pred.T])[..., None])
        res1 = self.evaluate(metric.result())

        self.assertEqual(res0, res1)


@test_combinations.run_all_keras_modes
class TestSparseCategoricalBoundaryAccuracy(test_combinations.TestCase):
    DIAG = np.array([
        [0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1],
        [2, 0, 0, 0, 0, 1, 1, 1, 1],
        [2, 2, 0, 0, 0, 0, 1, 1, 1],
        [2, 2, 2, 0, 0, 0, 0, 1, 1],
        [0, 2, 2, 2, 0, 0, 0, 0, 1],
        [0, 0, 2, 2, 2, 0, 0, 0, 0]]).astype('int32')

    def test_config(self):
        metric = SparseCategoricalBoundaryAccuracy(radius=1, name='metric1')
        self.assertEqual(metric.radius, 1)
        self.assertEqual(metric.name, 'metric1')

    def test_zeros(self):
        targets = np.zeros((2, 32, 32, 1), 'int32')
        probs = np.ones((2, 32, 32, 3), 'float32')
        probs[..., 0] = 1.

        metric = SparseCategoricalBoundaryAccuracy()
        metric.update_state(targets, probs)
        result = self.evaluate(metric.result())
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_ones(self):
        targets = np.ones((2, 32, 32, 1), 'int32')
        probs = np.zeros((2, 32, 32, 3), 'float32')
        probs[..., 1] = 1.

        metric = SparseCategoricalBoundaryAccuracy()
        metric.update_state(targets, probs)
        result = self.evaluate(metric.result())
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_false(self):
        pred = 1 - np.eye(3)[self.DIAG.reshape(-1)].reshape((1, 9, 9, 3))

        metric = SparseCategoricalBoundaryAccuracy()
        metric.update_state(self.DIAG[None, ..., None], pred)
        result = self.evaluate(metric.result())
        self.assertAlmostEqual(result, 0.0, places=7)

    def test_true(self):
        pred = np.eye(3)[self.DIAG.reshape(-1)].reshape((1, 9, 9, 3))

        metric = SparseCategoricalBoundaryAccuracy()
        metric.update_state(self.DIAG[None, ..., None], pred)
        result = self.evaluate(metric.result())
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_value(self):
        pred = np.eye(3)[self.DIAG.reshape(-1)].reshape((1, 9, 9, 3))
        pred[0, 4, 4] = [0, 1, 0]

        metric = SparseCategoricalBoundaryAccuracy()
        metric.update_state(self.DIAG[None, ..., None], pred)
        result = self.evaluate(metric.result())

        self.assertAlmostEqual(result, 0.9811321, places=6)  # 0.94117647 without frame

    def test_batch(self):
        pred = np.eye(3)[self.DIAG.reshape(-1)].reshape((1, 9, 9, 3))
        pred[0, 4, 4] = [0, 1, 0]

        metric = SparseCategoricalBoundaryAccuracy()
        metric.update_state(self.DIAG[None, ..., None], pred)
        metric.update_state(self.DIAG.T[None, ..., None], pred.transpose(0, 2, 1, 3))
        res0 = self.evaluate(metric.result())

        metric.reset_states()
        metric.update_state(
            np.stack([self.DIAG, self.DIAG.T])[..., None],
            np.concatenate([pred, pred.transpose(0, 2, 1, 3)], axis=0))
        res1 = self.evaluate(metric.result())

        self.assertEqual(res0, res1)


if __name__ == '__main__':
    tf.test.main()
