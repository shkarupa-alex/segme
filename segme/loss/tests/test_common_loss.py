import numpy as np
import tensorflow as tf
from keras.testing_infra import test_combinations
from segme.loss.common_loss import validate_input, to_logits, to_probs, to_1hot, mae, crossentropy, iou


@test_combinations.run_all_keras_modes
class TestUtils(test_combinations.TestCase):
    def test_validate_input(self):
        targets = (np.random.uniform(size=(2, 4, 4, 1)) > .5).astype('int32')
        probs = np.random.uniform(size=(2, 4, 4, 1))
        weights = np.random.uniform(size=(2, 4, 4, 1))

        y_true, y_pred, sample_weight = validate_input(targets, probs, weights, dtype=None, rank=None, channel=None)
        y_true, y_pred, sample_weight = self.evaluate([y_true, y_pred, sample_weight])
        self.assertAllClose(targets, y_true)
        self.assertAllClose(probs, y_pred)
        self.assertAllClose(weights, sample_weight)

    def test_to_logits(self):
        expected1 = tf.constant([
            [[[0.4250706654827763], [7.219920928747051], [7.14131948950217], [2.5576064452206024]],
             [[1.342442193620409], [0.20020616879804165], [3.977300484664198], [6.280817910206608]],
             [[0.3206719246447576], [3.0176225602425912], [2.902292891065069], [3.369106587128292]],
             [[2.6576544216404563], [6.863726154333165], [4.581314280496405], [7.433728759092233]]],
            [[[8.13888654097292], [8.311411218599392], [0.8372454481780323], [2.859455217953778]],
             [[2.0984725413538854], [4.619268334888168], [8.708732477440673], [1.9102341271004541]],
             [[3.4914178176388266], [4.551627675234152], [7.709902261544302], [3.3982255596983277]],
             [[0.9182162683255968], [3.0387004793287886], [2.1883984916630697], [1.3921544038795197]]]], 'float32')
        expected4 = tf.constant([
            [[[0.4250706654827763, 7.219920928747051, 1.14131948950217, 2.5576064452206024],
              [1.342442193620409, 0.20020616879804165, 6.977300484664198, 6.280817910206608]],
             [[0.3206719246447576, 0.0176225602425912, 1.902292891065069, 3.369106587128292],
              [2.6576544216404563, 1.863726154333165, 4.581314280496405, 7.433728759092233]],
             [[8.13888654097292, 1.311411218599392, 0.8372454481780323, 2.859455217953778],
              [2.0984725413538854, 4.619268334888168, 8.708732477440673, 1.9102341271004541]],
             [[3.4914178176388266, 4.551627675234152, 7.709902261544302, 3.3982255596983277],
              [0.9182162683255968, 7.0387004793287886, 2.1883984916630697, 1.3921544038795197]]]], 'float32')

        with self.assertRaisesRegex(ValueError, 'Unable to restore logits'):
            to_logits(tf.zeros((1, 2, 2, 1)), from_logits=False)

        with self.assertRaisesRegex(ValueError, 'Unable to restore logits'):
            to_logits(tf.zeros((1, 2, 2, 1)), from_logits=False)

        logits1 = to_logits(expected1, from_logits=True)
        logits1 = self.evaluate(logits1)
        self.assertAllClose(logits1, expected1, atol=1e-6)

        probs1 = tf.nn.sigmoid(expected1)
        probs1._keras_logits = tf.constant(expected1)
        logits1 = to_logits(probs1, from_logits=False)
        logits1 = self.evaluate(logits1)
        self.assertAllClose(logits1, expected1, atol=1e-6)

        probs4 = tf.nn.sigmoid(expected4)
        probs4._keras_logits = tf.constant(expected4)
        logits4 = to_logits(probs4, from_logits=False)
        logits4 = self.evaluate(logits4)
        self.assertAllClose(logits4, expected4, atol=1e-6)

        with self.assertRaisesRegex(ValueError, 'does not represent logits'):
            probs1 = tf.zeros((1, 2, 2, 1))
            probs1._keras_logits = tf.zeros((1, 2, 2, 1))
            to_logits(probs1, from_logits=True)

    def test_to_probs(self):
        logits1 = tf.constant([
            [[[0.4250706654827763], [7.219920928747051], [7.14131948950217], [2.5576064452206024]],
             [[1.342442193620409], [0.20020616879804165], [3.977300484664198], [6.280817910206608]],
             [[0.3206719246447576], [3.0176225602425912], [2.902292891065069], [3.369106587128292]],
             [[2.6576544216404563], [6.863726154333165], [4.581314280496405], [7.433728759092233]]],
            [[[8.13888654097292], [8.311411218599392], [0.8372454481780323], [2.859455217953778]],
             [[2.0984725413538854], [4.619268334888168], [8.708732477440673], [1.9102341271004541]],
             [[3.4914178176388266], [4.551627675234152], [7.709902261544302], [3.3982255596983277]],
             [[0.9182162683255968], [3.0387004793287886], [2.1883984916630697], [1.3921544038795197]]]], 'float32')
        logits4 = tf.constant([
            [[[0.4250706654827763, 7.219920928747051, 1.14131948950217, 2.5576064452206024],
              [1.342442193620409, 0.20020616879804165, 6.977300484664198, 6.280817910206608]],
             [[0.3206719246447576, 0.0176225602425912, 1.902292891065069, 3.369106587128292],
              [2.6576544216404563, 1.863726154333165, 4.581314280496405, 7.433728759092233]],
             [[8.13888654097292, 1.311411218599392, 0.8372454481780323, 2.859455217953778],
              [2.0984725413538854, 4.619268334888168, 8.708732477440673, 1.9102341271004541]],
             [[3.4914178176388266, 4.551627675234152, 7.709902261544302, 3.3982255596983277],
              [0.9182162683255968, 7.0387004793287886, 2.1883984916630697, 1.3921544038795197]]]], 'float32')

        probs1 = to_probs(logits1, from_logits=True, force_sigmoid=False)
        expected1 = tf.nn.sigmoid(logits1)
        probs1, expected1 = self.evaluate([probs1, expected1])
        self.assertAllClose(probs1, expected1, atol=1e-6)

        probs4 = to_probs(logits4, from_logits=True, force_sigmoid=False)
        expected4 = tf.nn.softmax(logits4)
        probs4, expected4 = self.evaluate([probs4, expected4])
        self.assertAllClose(probs4, expected4, atol=1e-6)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'does not represent probabilities'):
            to_probs(logits1, from_logits=False, force_sigmoid=False)

    def test_to_1hot(self):
        targets1 = tf.constant([
            [[[0], [0], [1], [0]], [[1], [0], [1], [1]], [[0], [1], [0], [1]], [[0], [1], [1], [1]]],
            [[[0], [1], [1], [0]], [[1], [0], [0], [1]], [[0], [1], [1], [0]], [[1], [1], [1], [1]]]], 'int32')
        targets4 = tf.constant([[[[1], [3]], [[3], [3]], [[1], [2]], [[2], [1]]]], 'int32')

        targets1h, _ = to_1hot(targets1, np.zeros((2, 4, 4, 1), 'float32'))
        expected1h = tf.concat([1 - targets1, targets1], axis=-1)
        targets1h, expected1h = self.evaluate([targets1h, expected1h])
        self.assertAllClose(targets1h, expected1h, atol=1e-6)

        targets4h, _ = to_1hot(targets4, np.zeros((2, 4, 4, 4), 'float32'))
        expected4h = tf.constant([
            [[[0, 1, 0, 0], [0, 0, 0, 1]], [[0, 0, 0, 1], [0, 0, 0, 1]],
             [[0, 1, 0, 0], [0, 0, 1, 0]], [[0, 0, 1, 0], [0, 1, 0, 0]]]], 'int32')
        targets4h, expected4h = self.evaluate([targets4h, expected4h])
        self.assertAllClose(targets4h, expected4h, atol=4e-6)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'Condition x == y did not hold'):
            to_1hot(targets1, np.ones((2, 4, 4, 1), 'float32') * 2.)


@test_combinations.run_all_keras_modes
class TestMAE(test_combinations.TestCase):
    def test_zeros(self):
        logits = tf.ones((1, 16, 16, 1), 'float32') * (-10.)
        targets = tf.zeros((1, 16, 16, 1), 'int32')

        result = mae(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAllClose(result, np.zeros_like(logits), atol=1e-4)

    def test_ones(self):
        logits = tf.ones((1, 16, 16, 1), 'float32') * 10.
        targets = tf.ones((1, 16, 16, 1), 'int32')

        result = mae(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAllClose(result, np.zeros_like(logits), atol=1e-4)

    def test_false(self):
        logits = tf.ones((1, 16, 16, 1), 'float32') * (-10.)
        targets = tf.ones((1, 16, 16, 1), 'int32')

        result = mae(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAllClose(result, np.ones_like(logits), atol=1e-2)

    def test_true(self):
        logits = tf.ones((1, 16, 16, 1), 'float32') * 10.
        targets = tf.zeros((1, 16, 16, 1), 'int32')

        result = mae(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAllClose(result, np.ones_like(logits), atol=1e-2)

    def test_multi(self):
        logits = tf.constant([
            [[[0.4250706654827763, -7.219920928747051, -1.14131948950217, 2.5576064452206024],
              [-1.342442193620409, 0.20020616879804165, -6.977300484664198, 6.280817910206608]],
             [[0.3206719246447576, 0.0176225602425912, -1.902292891065069, -3.369106587128292],
              [-2.6576544216404563, 1.863726154333165, 4.581314280496405, -7.433728759092233]],
             [[8.13888654097292, 1.311411218599392, 0.8372454481780323, -2.859455217953778],
              [-2.0984725413538854, -4.619268334888168, 8.708732477440673, 1.9102341271004541]],
             [[3.4914178176388266, -4.551627675234152, 7.709902261544302, 3.3982255596983277],
              [-0.9182162683255968, -7.0387004793287886, 2.1883984916630697, 1.3921544038795197]]]], 'float32')
        targets = tf.constant([[[[1], [3]], [[3], [3]], [[1], [2]], [[2], [1]]]], 'int32')

        result = mae(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAlmostEqual(result.mean(), 0.5163353, places=7)

    def test_value(self):
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

        result = mae(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAlmostEqual(result.mean(), 0.40374124, places=6)

    def test_weight(self):
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

        result = self.evaluate(mae(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True))
        self.assertAlmostEqual(result.mean(), 0.40374124, places=6)

        result = self.evaluate(mae(
            y_true=targets[:, :, :2, :], y_pred=logits[:, :, :2, :], sample_weight=None, from_logits=True))
        self.assertAlmostEqual(result.mean(), 0.45837218, places=7)

        result = self.evaluate(mae(
            y_true=targets, y_pred=logits, sample_weight=weights, from_logits=True))
        self.assertAlmostEqual(result.mean(), 0.22918609, places=7)

        result = self.evaluate(mae(
            y_true=targets, y_pred=logits, sample_weight=weights * 2, from_logits=True))
        self.assertAlmostEqual(result.mean(), 0.22918609 * 2., places=6)


@test_combinations.run_all_keras_modes
class TestCrossentropy(test_combinations.TestCase):
    def test_zeros(self):
        logits = tf.ones((1, 16, 16, 1), 'float32') * (-10.)
        targets = tf.zeros((1, 16, 16, 1), 'int32')

        result = crossentropy(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAllClose(result, np.zeros_like(logits), atol=1e-4)

    def test_ones(self):
        logits = tf.ones((1, 16, 16, 1), 'float32') * 10.
        targets = tf.ones((1, 16, 16, 1), 'int32')

        result = crossentropy(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAllClose(result, np.zeros_like(logits), atol=1e-4)

    def test_false(self):
        logits = tf.ones((1, 16, 16, 1), 'float32') * (-10.)
        targets = tf.ones((1, 16, 16, 1), 'int32')

        result = crossentropy(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAllClose(result, np.ones_like(logits) * 10., atol=1e-2)

    def test_true(self):
        logits = tf.ones((1, 16, 16, 1), 'float32') * 10.
        targets = tf.zeros((1, 16, 16, 1), 'int32')

        result = crossentropy(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAllClose(result, np.ones_like(logits) * 10., atol=1e-2)

    def test_multi(self):
        logits = tf.constant([
            [[[0.4250706654827763, -7.219920928747051, -1.14131948950217, 2.5576064452206024],
              [-1.342442193620409, 0.20020616879804165, -6.977300484664198, 6.280817910206608]],
             [[0.3206719246447576, 0.0176225602425912, -1.902292891065069, -3.369106587128292],
              [-2.6576544216404563, 1.863726154333165, 4.581314280496405, -7.433728759092233]],
             [[8.13888654097292, 1.311411218599392, 0.8372454481780323, -2.859455217953778],
              [-2.0984725413538854, -4.619268334888168, 8.708732477440673, 1.9102341271004541]],
             [[3.4914178176388266, -4.551627675234152, 7.709902261544302, 3.3982255596983277],
              [-0.9182162683255968, -7.0387004793287886, 2.1883984916630697, 1.3921544038795197]]]], 'float32')
        targets = tf.constant([[[[1], [3]], [[3], [3]], [[1], [2]], [[2], [1]]]], 'int32')

        result = crossentropy(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAlmostEqual(result.mean(), 5.34982, places=5)

    def test_value(self):
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

        result = crossentropy(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAlmostEqual(result.mean(), 1.5985572, places=6)

    def test_weight(self):
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

        result = self.evaluate(crossentropy(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True))
        self.assertAlmostEqual(result.mean(), 1.5985572, places=6)

        result = self.evaluate(crossentropy(
            y_true=targets[:, :, :2, :], y_pred=logits[:, :, :2, :], sample_weight=None, from_logits=True))
        self.assertAlmostEqual(result.mean(), 1.8511493, places=7)

        result = self.evaluate(crossentropy(y_true=targets, y_pred=logits, sample_weight=weights, from_logits=True))
        self.assertAlmostEqual(result.mean(), 0.9255747, places=7)

        result = self.evaluate(crossentropy(y_true=targets, y_pred=logits, sample_weight=weights * 2, from_logits=True))
        self.assertAlmostEqual(result.mean(), 0.9255747 * 2., places=6)


@test_combinations.run_all_keras_modes
class TestIOU(test_combinations.TestCase):
    def test_zeros(self):
        logits = tf.ones((1, 16, 16, 1), 'float32') * (-10.)
        targets = tf.zeros((1, 16, 16, 1), 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=False)
        result = self.evaluate(result)
        self.assertAllClose(result, np.zeros_like(logits), atol=1e-2)

    def test_ones(self):
        logits = tf.ones((1, 16, 16, 1), 'float32') * 10.
        targets = tf.ones((1, 16, 16, 1), 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=False)
        result = self.evaluate(result)
        self.assertAllClose(result, np.zeros_like(logits), atol=1e-2)

    def test_false(self):
        logits = tf.ones((1, 16, 16, 1), 'float32') * (-10.)
        targets = tf.ones((1, 16, 16, 1), 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=False)
        result = self.evaluate(result)
        self.assertAllClose(result, np.ones_like(logits), atol=1e-2)

    def test_true(self):
        logits = tf.ones((1, 16, 16, 1), 'float32') * 10.
        targets = tf.zeros((1, 16, 16, 1), 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=False)
        result = self.evaluate(result)
        self.assertAllClose(result, np.ones_like(logits), atol=1e-2)

    def test_multi(self):
        logits = tf.constant([
            [[[0.4250706654827763, -7.219920928747051, -1.14131948950217, 2.5576064452206024],
              [-1.342442193620409, 0.20020616879804165, -6.977300484664198, 6.280817910206608]],
             [[0.3206719246447576, 0.0176225602425912, -1.902292891065069, -3.369106587128292],
              [-2.6576544216404563, 1.863726154333165, 4.581314280496405, -7.433728759092233]],
             [[8.13888654097292, 1.311411218599392, 0.8372454481780323, -2.859455217953778],
              [-2.0984725413538854, -4.619268334888168, 8.708732477440673, 1.9102341271004541]],
             [[3.4914178176388266, -4.551627675234152, 7.709902261544302, 3.3982255596983277],
              [-0.9182162683255968, -7.0387004793287886, 2.1883984916630697, 1.3921544038795197]]]], 'float32')
        targets = tf.constant([[[[1], [3]], [[3], [3]], [[1], [2]], [[2], [1]]]], 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=False)
        result = self.evaluate(result)
        self.assertAlmostEqual(result.mean(), 0.6096517, places=7)

    def test_value(self):
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

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=False)
        result = self.evaluate(result)
        self.assertAlmostEqual(result.mean(), 0.48672712, places=6)

    def test_weight(self):
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

        result = self.evaluate(iou(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=False))
        self.assertAlmostEqual(result.mean(), 0.48672712, places=6)

        result = self.evaluate(iou(
            y_true=targets[:, :, :2, :], y_pred=logits[:, :, :2, :], sample_weight=None, from_logits=True, dice=False))
        self.assertAlmostEqual(result.mean(), 0.49471125, places=7)

        result = self.evaluate(iou(
            y_true=targets, y_pred=logits, sample_weight=weights, from_logits=True, dice=False))
        self.assertAlmostEqual(result.mean(), 0.27417138, places=7)

        result = self.evaluate(iou(
            y_true=targets, y_pred=logits, sample_weight=weights * 2, from_logits=True, dice=False))
        self.assertAlmostEqual(result.mean(), 0.27417138 * 2., places=6)


@test_combinations.run_all_keras_modes
class TestDice(test_combinations.TestCase):
    def test_zeros(self):
        logits = tf.ones((1, 16, 16, 1), 'float32') * (-10)
        targets = tf.zeros((1, 16, 16, 1), 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=True)
        result = self.evaluate(result)
        self.assertAllClose(result, np.zeros_like(logits), atol=1e-2)

    def test_ones(self):
        logits = tf.ones((1, 16, 16, 1), 'float32') * 10.
        targets = tf.ones((1, 16, 16, 1), 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=True)
        result = self.evaluate(result)
        self.assertAllClose(result, np.zeros_like(logits), atol=1e-2)

    def test_false(self):
        logits = tf.ones((1, 16, 16, 1), 'float32') * (-10.)
        targets = tf.ones((1, 16, 16, 1), 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=True)
        result = self.evaluate(result)
        self.assertAllClose(result, np.ones_like(logits), atol=1e-2)

    def test_true(self):
        logits = tf.ones((1, 16, 16, 1), 'float32') * 10.
        targets = tf.zeros((1, 16, 16, 1), 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=True)
        result = self.evaluate(result)
        self.assertAllClose(result, np.ones_like(logits), atol=1e-2)

    def test_multi(self):
        logits = tf.constant([
            [[[0.4250706654827763, -7.219920928747051, -1.14131948950217, 2.5576064452206024],
              [-1.342442193620409, 0.20020616879804165, -6.977300484664198, 6.280817910206608]],
             [[0.3206719246447576, 0.0176225602425912, -1.902292891065069, -3.369106587128292],
              [-2.6576544216404563, 1.863726154333165, 4.581314280496405, -7.433728759092233]],
             [[8.13888654097292, 1.311411218599392, 0.8372454481780323, -2.859455217953778],
              [-2.0984725413538854, -4.619268334888168, 8.708732477440673, 1.9102341271004541]],
             [[3.4914178176388266, -4.551627675234152, 7.709902261544302, 3.3982255596983277],
              [-0.9182162683255968, -7.0387004793287886, 2.1883984916630697, 1.3921544038795197]]]], 'float32')
        targets = tf.constant([[[[1], [3]], [[3], [3]], [[1], [2]], [[2], [1]]]], 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=True)
        result = self.evaluate(result)
        self.assertAlmostEqual(result.mean(), 0.60636544, places=6)

    def test_value(self):
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

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=True)
        result = self.evaluate(result)
        self.assertAlmostEqual(result.mean(), 0.46608606, places=6)

    def test_weight(self):
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

        result = self.evaluate(iou(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=True))
        self.assertAlmostEqual(result.mean(), 0.46608606, places=6)

        result = self.evaluate(iou(
            y_true=targets[:, :, :2, :], y_pred=logits[:, :, :2, :], sample_weight=None, from_logits=True, dice=True))
        self.assertAlmostEqual(result.mean(), 0.4710906, places=6)

        result = self.evaluate(iou(
            y_true=targets, y_pred=logits, sample_weight=weights, from_logits=True, dice=True))
        self.assertAlmostEqual(result.mean(), 0.26115888, places=6)

        result = self.evaluate(iou(
            y_true=targets, y_pred=logits, sample_weight=weights * 2, from_logits=True, dice=True))
        self.assertAlmostEqual(result.mean(), 0.26115888 * 2., places=6)


if __name__ == '__main__':
    tf.test.main()
