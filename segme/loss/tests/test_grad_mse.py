import cv2
import numpy as np
import tensorflow as tf
from keras import keras_parameterized, layers, models
from keras.utils.losses_utils import ReductionV2 as Reduction
from ..grad_mse import GradientMeanSquaredError
from ..grad_mse import gradient_mean_squared_error


@keras_parameterized.run_all_keras_modes
class TestGradientMeanSquaredError(keras_parameterized.TestCase):
    def test_config(self):
        bce_obj = GradientMeanSquaredError(
            reduction=Reduction.NONE,
            name='loss1'
        )
        self.assertEqual(bce_obj.name, 'loss1')
        self.assertEqual(bce_obj.reduction, Reduction.NONE)

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

        result = gradient_mean_squared_error(y_true=targets, y_pred=probs, sigma=1.4)
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

        result = gradient_mean_squared_error(y_true=targets[None, ..., None], y_pred=probs[None, ..., None], sigma=1.4)
        result = np.sum(self.evaluate(result) * trim[None, ...]).item()
        self.assertAlmostEqual(result, 1.667233, places=5)  # same for reduce_sum

    def test_weight_4d(self):
        logits = tf.constant([
            [[[0.4250706654827763], [7.219920928747051], [7.14131948950217], [2.5576064452206024]],
             [[1.342442193620409], [0.20020616879804165], [3.977300484664198], [6.280817910206608]],
             [[0.3206719246447576], [3.0176225602425912], [2.902292891065069], [3.369106587128292]],
             [[2.6576544216404563], [6.863726154333165], [4.581314280496405], [7.433728759092233]]],
            [[[8.13888654097292], [8.311411218599392], [0.8372454481780323], [2.859455217953778]],
             [[2.0984725413538854], [4.619268334888168], [8.708732477440673], [1.9102341271004541]],
             [[3.4914178176388266], [4.551627675234152], [7.709902261544302], [3.3982255596983277]],
             [[0.9182162683255968], [3.0387004793287886], [2.1883984916630697], [1.3921544038795197]]]], 'float32')
        targets = tf.constant([
            [[[0], [0], [1], [0]], [[1], [0], [1], [1]], [[0], [1], [0], [1]], [[0], [1], [1], [1]]],
            [[[0], [1], [1], [0]], [[1], [0], [0], [1]], [[0], [1], [1], [0]], [[1], [1], [1], [1]]]], 'int32')
        weights = tf.concat([tf.ones((2, 4, 2, 1)), tf.zeros((2, 4, 2, 1))], axis=2)

        loss = GradientMeanSquaredError(reduction=Reduction.SUM)

        result = self.evaluate(loss(targets, logits)).item()
        self.assertAlmostEqual(result, 8.369307518005371, places=7)

        result = self.evaluate(loss(targets, logits, weights)).item()
        self.assertAlmostEqual(result, 1.836201548576355, places=6)

    def test_keras_model_compile(self):
        model = models.Sequential([
            layers.Input(shape=(100,)),
            layers.Dense(5)]
        )
        model.compile(loss='SegMe>gradient_mean_squared_error')


if __name__ == '__main__':
    tf.test.main()
