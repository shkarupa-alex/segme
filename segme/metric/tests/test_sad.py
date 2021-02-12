import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from ..sad import SAD


@keras_parameterized.run_all_keras_modes
class TestSAD(keras_parameterized.TestCase):
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
        metric = SAD(
            divider=2.,
            name='metric1'
        )
        self.assertEqual(metric.divider, 2.)
        self.assertEqual(metric.name, 'metric1')

    def test_value(self):
        trim = np.where(cv2.dilate(self.SNAKE, np.ones((2, 2), 'float32')) > 0, 1., 0.)
        pred = np.round((self.SNAKE / 128.) ** 2 * 255. / 3.97)

        metric = SAD()
        metric.update_state(self.SNAKE[None, ..., None], pred[None, ..., None], trim[None, ..., None])
        result = self.evaluate(metric.result())

        self.assertAlmostEqual(result, 6.8941178, places=7)


if __name__ == '__main__':
    tf.test.main()
