import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from ..grad_mse import GradientMeanSquaredError
from ..grad_mse import gradient_mean_squared_error


@keras_parameterized.run_all_keras_modes
class TestGradientMeanSquaredError(keras_parameterized.TestCase):
    def test_config(self):
        bce_obj = GradientMeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE,
            name='loss1'
        )
        self.assertEqual(bce_obj.name, 'loss1')
        self.assertEqual(bce_obj.reduction, tf.keras.losses.Reduction.NONE)

    def test_zeros(self):
        probs = tf.constant([[
            [[0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0]],
        ]], 'float32')
        targets = tf.constant([[
            [[0], [0], [0]],
            [[0], [0], [0]],
            [[0], [0], [0]],
        ]], 'int32')

        result = gradient_mean_squared_error(y_true=targets, y_pred=probs)
        result = self.evaluate(result).tolist()

        self.assertAllClose(result, [[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]])

    def test_value_4d(self):
        # Very simple loss, not checked with found reimplementation
        targets = np.round(np.array([
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
        probs = np.round((targets / 128.) ** 2 * 255. / 3.97)
        trim = np.where(cv2.dilate(targets, np.ones((2, 2), 'float32')) > 0, 1., 0.)

        result = gradient_mean_squared_error(y_true=targets[None, ..., None], y_pred=probs[None, ..., None])
        result = np.sum(self.evaluate(result) * trim[None, ...]).item()
        self.assertAlmostEqual(result, 1.667233, places=5)  # same for reduce_sum

    def test_keras_model_compile(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(100,)),
            tf.keras.layers.Dense(5)]
        )
        model.compile(loss='SegMe>gradient_mean_squared_error')


if __name__ == '__main__':
    tf.test.main()
