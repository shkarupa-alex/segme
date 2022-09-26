import cv2
import numpy as np
import tensorflow as tf
from keras.testing_infra import test_combinations
from segme.metric.matting.mse import MSE


@test_combinations.run_all_keras_modes
class TestMSE(test_combinations.TestCase):
    SNAKE = np.array([
        [1, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 4, 5, 6, 0, 0, 0, 0],
        [0, 0, 0, 0, 7, 8, 9, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 7, 0],
        [0, 2, 1, 2, 3, 4, 5, 6, 0],
        [0, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 0, 6, 5, 4, 3, 2, 1],
        [0, 5, 0, 0, 0, 0, 0, 0, 2],
        [0, 6, 7, 8, 9, 8, 7, 0, 3],
        [0, 0, 0, 0, 0, 0, 7, 5, 4]
    ]).astype('float32') / 9.

    def test_config(self):
        metric = MSE(name='metric1')
        self.assertEqual(metric.name, 'metric1')

    def test_zeros(self):
        targets = np.zeros((2, 32, 32, 1), 'int32')
        probs = np.zeros((2, 32, 32, 1), 'float32')
        weight = np.ones((2, 32, 32, 1), 'float32')

        metric = MSE()
        metric.update_state(targets, probs, weight)
        result = self.evaluate(metric.result())
        self.assertAlmostEqual(result, 0.0, places=7)

    def test_value(self):
        trim = np.where(cv2.dilate(self.SNAKE, np.ones((2, 2), 'float32')) > 0, 1., 0.)
        pred = (self.SNAKE * 1.9921875) ** 2 / 3.97

        metric = MSE()
        metric.update_state(self.SNAKE[None, ..., None], pred[None, ..., None], trim[None, ..., None])
        result = self.evaluate(metric.result())

        self.assertAlmostEqual(result, 20.319115, places=5)

    def test_unweighted(self):
        pred = (self.SNAKE * 1.9921875) ** 2 / 3.97

        metric = MSE()
        metric.update_state(self.SNAKE[None, ..., None], pred[None, ..., None])
        result = self.evaluate(metric.result())

        self.assertAlmostEqual(result, 15.803756, places=6)

    def test_batch(self):
        trim0 = np.where(cv2.dilate(self.SNAKE, np.ones((2, 2), 'float32')) > 0, 1., 0.)
        pred0 = (self.SNAKE * 1.9921875) ** 2 / 3.97

        targ1 = np.pad(self.SNAKE[3:, 3:], [[0, 3], [0, 3]])
        trim1 = np.pad(trim0[3:, 3:], [[0, 3], [0, 3]])
        pred1 = np.pad(pred0[3:, 3:], [[0, 3], [0, 3]])

        metric = MSE()
        metric.update_state(self.SNAKE[None, ..., None], pred0[None, ..., None], trim0[None, ..., None])
        metric.update_state(targ1[None, ..., None], pred1[None, ..., None], trim1[None, ..., None])
        res0 = self.evaluate(metric.result())

        metric.reset_states()
        metric.update_state(
            np.array([self.SNAKE[..., None], targ1[..., None]]),
            np.array([pred0[..., None], pred1[..., None]]),
            np.array([trim0[..., None], trim1[..., None]]))
        res1 = self.evaluate(metric.result())

        self.assertAlmostEqual(res0, res1, places=5)


if __name__ == '__main__':
    tf.test.main()
