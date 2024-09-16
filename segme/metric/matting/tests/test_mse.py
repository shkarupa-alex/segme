import cv2
import numpy as np
from keras.src import backend
from keras.src import ops
from keras.src import testing

from segme.metric.matting.mse import MSE


class TestMSE(testing.TestCase):
    SNAKE = (
        np.array(
            [
                [1, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 3, 4, 5, 6, 0, 0, 0, 0],
                [0, 0, 0, 0, 7, 8, 9, 8, 0],
                [0, 0, 0, 0, 0, 0, 0, 7, 0],
                [0, 2, 1, 2, 3, 4, 5, 6, 0],
                [0, 3, 0, 0, 0, 0, 0, 0, 0],
                [0, 4, 0, 6, 5, 4, 3, 2, 1],
                [0, 5, 0, 0, 0, 0, 0, 0, 2],
                [0, 6, 7, 8, 9, 8, 7, 0, 3],
                [0, 0, 0, 0, 0, 0, 7, 5, 4],
            ]
        ).astype("float32")
        / 9.0
    )

    def test_config(self):
        metric = MSE(name="metric1")
        self.assertEqual(metric.name, "metric1")

    def test_zeros(self):
        targets = ops.zeros((2, 32, 32, 1), "int32")
        probs = ops.zeros((2, 32, 32, 1), "float32")
        weight = ops.ones((2, 32, 32, 1), "float32")

        metric = MSE()
        metric.update_state(targets, probs, weight)
        self.assertAlmostEqual(metric.result(), 0.0, decimal=7)

    def test_value(self):
        trim = np.where(
            cv2.dilate(self.SNAKE, np.ones((2, 2), "float32")) > 0, 1.0, 0.0
        )
        pred = (self.SNAKE * 1.9921875) ** 2 / 3.97

        metric = MSE()
        metric.update_state(
            self.SNAKE[None, ..., None],
            pred[None, ..., None],
            trim[None, ..., None],
        )
        self.assertAlmostEqual(metric.result(), 20.319115, decimal=5)

    def test_unweighted(self):
        pred = (self.SNAKE * 1.9921875) ** 2 / 3.97

        metric = MSE()
        metric.update_state(self.SNAKE[None, ..., None], pred[None, ..., None])
        self.assertAlmostEqual(metric.result(), 15.803756, decimal=6)

    def test_batch(self):
        trim0 = np.where(
            cv2.dilate(self.SNAKE, np.ones((2, 2), "float32")) > 0, 1.0, 0.0
        )
        pred0 = (self.SNAKE * 1.9921875) ** 2 / 3.97

        targ1 = np.pad(self.SNAKE[3:, 3:], [[0, 3], [0, 3]])
        trim1 = np.pad(trim0[3:, 3:], [[0, 3], [0, 3]])
        pred1 = np.pad(pred0[3:, 3:], [[0, 3], [0, 3]])

        metric = MSE()
        metric.update_state(
            self.SNAKE[None, ..., None],
            pred0[None, ..., None],
            trim0[None, ..., None],
        )
        metric.update_state(
            targ1[None, ..., None],
            pred1[None, ..., None],
            trim1[None, ..., None],
        )
        res0 = backend.convert_to_numpy(metric.result())

        metric.reset_state()
        metric.update_state(
            np.array([self.SNAKE[..., None], targ1[..., None]]),
            np.array([pred0[..., None], pred1[..., None]]),
            np.array([trim0[..., None], trim1[..., None]]),
        )
        res1 = metric.result()

        self.assertAlmostEqual(res0, res1, decimal=5)
