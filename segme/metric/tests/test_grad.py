import cv2
import numpy as np
import tensorflow as tf
from keras import keras_parameterized
from ..grad import Grad


@keras_parameterized.run_all_keras_modes
class TestGrad(keras_parameterized.TestCase):
    SNAKE = np.round(np.array([
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
    ]).astype('float32') * 255. / 9.)

    def test_config(self):
        metric = Grad(
            divider=2.,
            name='metric1'
        )
        self.assertEqual(metric.divider, 2.)
        self.assertEqual(metric.name, 'metric1')

    def test_zeros(self):
        targets = np.zeros((2, 32, 32, 1), 'int32')
        probs = np.zeros((2, 32, 32, 1), 'float32')
        weight = np.ones((2, 32, 32, 1), 'float32')

        metric = Grad()
        metric.update_state(targets, probs, weight)
        result = self.evaluate(metric.result())
        self.assertAlmostEqual(result, 0.0, places=7)

    def test_value(self):
        trim = np.where(cv2.dilate(self.SNAKE, np.ones((2, 2), 'float32')) > 0, 1., 0.)
        pred = np.round((self.SNAKE / 128.) ** 2 * 255. / 3.97)

        metric = Grad()
        metric.update_state(self.SNAKE[None, ..., None], pred[None, ..., None], trim[None, ..., None])
        result = self.evaluate(metric.result())

        self.assertAlmostEqual(result, 0.001667233, places=9)

    def test_unweighted(self):
        pred = np.round((self.SNAKE / 128.) ** 2 * 255. / 3.97)

        metric = Grad()
        metric.update_state(self.SNAKE[None, ..., None], pred[None, ..., None])
        result = self.evaluate(metric.result())

        self.assertAlmostEqual(result, 0.00253742, places=9)

    def test_batch(self):
        trim0 = np.where(cv2.dilate(self.SNAKE, np.ones((2, 2), 'float32')) > 0, 1., 0.)
        pred0 = np.round((self.SNAKE / 128.) ** 2 * 255. / 3.97)

        targ1 = np.pad(self.SNAKE[3:, 3:], [[0, 3], [0, 3]])
        trim1 = np.pad(trim0[3:, 3:], [[0, 3], [0, 3]])
        pred1 = np.pad(pred0[3:, 3:], [[0, 3], [0, 3]])

        metric = Grad()
        metric.update_state(self.SNAKE[None, ..., None], pred0[None, ..., None], trim0[None, ..., None])
        metric.update_state(targ1[None, ..., None], pred1[None, ..., None], trim1[None, ..., None])
        res0 = self.evaluate(metric.result())

        metric.reset_states()
        metric.update_state(
            np.array([self.SNAKE[..., None], targ1[..., None]]),
            np.array([pred0[..., None], pred1[..., None]]),
            np.array([trim0[..., None], trim1[..., None]]))
        res1 = self.evaluate(metric.result())

        self.assertEqual(res0, res1)

    def test_channel3(self):
        targets = np.zeros((2, 32, 32, 3), 'int32')
        probs = np.zeros((2, 32, 32, 3), 'float32')
        weight = np.ones((2, 32, 32, 3), 'float32')

        metric = Grad()
        metric.update_state(targets, probs, weight)
        result = self.evaluate(metric.result())
        self.assertAlmostEqual(result, 0.0, places=7)


if __name__ == '__main__':
    tf.test.main()
