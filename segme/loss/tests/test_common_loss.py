import numpy as np
import tensorflow as tf
from tf_keras.src.testing_infra import test_combinations
from segme.loss.common_loss import validate_input, to_logits, to_probs, to_1hot, weighted_loss, compute_gradient, \
    mae, mse, crossentropy, iou


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

    def test_weighted_loss(self):
        loss = np.random.uniform(size=(2, 4, 5, 3))
        weight = np.random.uniform(size=(2, 4, 5, 1)) - 0.5
        weight[weight < 0.] = 0.
        expected = np.array([
            (loss[0] * weight[0])[(weight[0] > 0.).repeat(loss.shape[-1], axis=-1)].mean(),
            (loss[1] * weight[1])[(weight[1] > 0.).repeat(loss.shape[-1], axis=-1)].mean()
        ])
        loss, weight = tf.constant(loss), tf.constant(weight)

        result = weighted_loss(loss, weight)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

    def test_compute_gradient(self):
        inputs = tf.constant([
            [[[0.], [0.], [1.], [0.]], [[1.], [0.], [1.], [1.]], [[0.], [1.], [0.], [1.]], [[0.], [1.], [1.], [1.]]],
            [[[0.], [1.], [1.], [0.]], [[1.], [0.], [0.], [1.]], [[0.], [1.], [1.], [0.]], [[1.], [1.], [1.], [1.]]]],
            'float32')
        expected_1sub = [
            [[[1.], [0.], [0.], [1.]], [[-1.], [1.], [-1.], [0.]], [[0.], [0.], [1.], [0.]]],
            [[[1.], [-1.], [-1.], [1.]], [[-1.], [1.], [1.], [-1.]], [[1.], [0.], [0.], [1.]]]]
        expected_2min = [
            [[[0.], [0.], [0.]], [[0.], [0.], [1.]], [[0.], [0.], [0.]], [[0.], [1.], [1.]]],
            [[[0.], [1.], [0.]], [[0.], [0.], [0.]], [[0.], [1.], [0.]], [[1.], [1.], [1.]]]]

        grad_1sub = compute_gradient(inputs, 1, 'sub')
        grad_1sub = self.evaluate(grad_1sub)
        self.assertAllClose(grad_1sub, expected_1sub)

        grad_2min = compute_gradient(inputs, 2, 'min')
        grad_2min = self.evaluate(grad_2min)
        self.assertAllClose(grad_2min, expected_2min)


@test_combinations.run_all_keras_modes
class TestMAE(test_combinations.TestCase):
    def test_zeros(self):
        logits = -10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'int32')

        result = mae(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_ones(self):
        logits = 10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'int32')

        result = mae(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_false(self):
        logits = -10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'int32')

        result = mae(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [1.] * 3, atol=6e-3)

    def test_true(self):
        logits = 10 * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'int32')

        result = mae(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [1.] * 3, atol=6e-3)

    def test_value(self):
        result = mae(
            y_true=BINARY_TARGETS, y_pred=BINARY_LOGITS, sample_weight=None, from_logits=True, regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.375533, 0.417319])

        result = mae(
            y_true=tf.cast(BINARY_TARGETS, 'float32'), y_pred=tf.nn.sigmoid(BINARY_LOGITS), sample_weight=None,
            from_logits=False, regression=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.375533, 0.417319])

    def test_weight(self):
        result = mae(
            y_true=BINARY_TARGETS[:, :, :2], y_pred=BINARY_LOGITS[:, :, :2], sample_weight=None, from_logits=True,
            regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.49504662, 0.17893231])

        result = mae(
            y_true=BINARY_TARGETS, y_pred=BINARY_LOGITS, sample_weight=BINARY_WEIGHTS, from_logits=True,
            regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.49504662, 0.17893231])

        result = mae(
            y_true=BINARY_TARGETS, y_pred=BINARY_LOGITS, sample_weight=BINARY_WEIGHTS * 2, from_logits=True,
            regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.99009323, 0.35786462])

    def test_multi(self):
        result = mae(y_true=MULTI_TARGETS, y_pred=MULTI_LOGITS, sample_weight=None, from_logits=True, regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.5163353])


@test_combinations.run_all_keras_modes
class TestMSE(test_combinations.TestCase):
    def test_zeros(self):
        logits = -10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'int32')

        result = mse(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_ones(self):
        logits = 10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'int32')

        result = mse(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_false(self):
        logits = -10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'int32')

        result = mse(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [1.] * 3, atol=6e-3)

    def test_true(self):
        logits = 10 * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'int32')

        result = mse(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [1.] * 3, atol=6e-3)

    def test_value(self):
        result = mse(y_true=BINARY_TARGETS, y_pred=BINARY_LOGITS, sample_weight=None, from_logits=True,
                     regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.30168968, 0.35166395])

        result = mse(
            y_true=tf.cast(BINARY_TARGETS, 'float32'), y_pred=tf.nn.sigmoid(BINARY_LOGITS), sample_weight=None,
            from_logits=False, regression=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.30168968, 0.35166395])

    def test_weight(self):
        result = mse(
            y_true=BINARY_TARGETS[:, :, :2], y_pred=BINARY_LOGITS[:, :, :2], sample_weight=None, from_logits=True,
            regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.3698082, 0.12967442])

        result = mse(
            y_true=BINARY_TARGETS, y_pred=BINARY_LOGITS, sample_weight=BINARY_WEIGHTS, from_logits=True,
            regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.3698082, 0.12967442])

        result = mse(
            y_true=BINARY_TARGETS, y_pred=BINARY_LOGITS, sample_weight=BINARY_WEIGHTS * 2, from_logits=True,
            regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.7396164, 0.25934884])

    def test_multi(self):
        result = mse(y_true=MULTI_TARGETS, y_pred=MULTI_LOGITS, sample_weight=None, from_logits=True, regression=False)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.42551923])


@test_combinations.run_all_keras_modes
class TestCrossentropy(test_combinations.TestCase):
    def test_zeros(self):
        logits = -10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'int32')

        result = crossentropy(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, force_binary=False, label_smoothing=0.)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_ones(self):
        logits = 10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'int32')

        result = crossentropy(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, force_binary=False, label_smoothing=0.)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_false(self):
        logits = -10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'int32')

        result = crossentropy(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, force_binary=False, label_smoothing=0.)
        result = self.evaluate(result)
        self.assertAllClose(result, [10.] * 3, atol=6e-3)

    def test_true(self):
        logits = 10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'int32')

        result = crossentropy(
            y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, force_binary=False, label_smoothing=0.)
        result = self.evaluate(result)
        self.assertAllClose(result, [10.] * 3, atol=6e-3)

    def test_value(self):
        result = crossentropy(
            y_true=BINARY_TARGETS, y_pred=BINARY_LOGITS, sample_weight=None, from_logits=True, force_binary=False,
            label_smoothing=0.)
        result = self.evaluate(result)
        self.assertAllClose(result, [1.2658163, 1.8140206])

    def test_value_smooth(self):
        result = crossentropy(
            y_true=BINARY_TARGETS, y_pred=BINARY_LOGITS, sample_weight=None, from_logits=True, force_binary=False,
            label_smoothing=0.05)
        result = self.evaluate(result)
        self.assertAllClose(result, [1.3035736, 1.8281653])

    def test_weight(self):
        result = crossentropy(
            y_true=BINARY_TARGETS[:, :, :2], y_pred=BINARY_LOGITS[:, :, :2], sample_weight=None, force_binary=False,
            from_logits=True, label_smoothing=0.)
        result = self.evaluate(result)
        self.assertAllClose(result, [1.6474432, 0.50508237])

        result = crossentropy(
            y_true=BINARY_TARGETS, y_pred=BINARY_LOGITS, sample_weight=BINARY_WEIGHTS, from_logits=True,
            force_binary=False, label_smoothing=0.)
        result = self.evaluate(result)
        self.assertAllClose(result, [1.6474432, 0.50508237])

        result = crossentropy(
            y_true=BINARY_TARGETS, y_pred=BINARY_LOGITS, sample_weight=BINARY_WEIGHTS * 2, from_logits=True,
            force_binary=False, label_smoothing=0.)
        result = self.evaluate(result)
        self.assertAllClose(result, [3.2948864, 1.0101647])

    def test_multi(self):
        result = crossentropy(
            y_true=MULTI_TARGETS, y_pred=MULTI_LOGITS, sample_weight=None, from_logits=True, force_binary=False,
            label_smoothing=0.)
        result = self.evaluate(result)
        self.assertAllClose(result, [5.34982])

    def test_multi_binary(self):
        result = crossentropy(
            y_true=MULTI_TARGETS, y_pred=MULTI_LOGITS, sample_weight=None, from_logits=True, force_binary=True,
            label_smoothing=0.)
        result = self.evaluate(result)
        self.assertAllClose(result, [7.669404])

    def test_multi_smooth(self):
        result = crossentropy(
            y_true=MULTI_TARGETS, y_pred=MULTI_LOGITS, sample_weight=None, from_logits=True, force_binary=False,
            label_smoothing=0.05)
        result = self.evaluate(result)
        self.assertAllClose(result, [5.34137])

    def test_multi_binary_smooth(self):
        result = crossentropy(
            y_true=MULTI_TARGETS, y_pred=MULTI_LOGITS, sample_weight=None, from_logits=True, force_binary=True,
            label_smoothing=0.05)
        result = self.evaluate(result)
        self.assertAllClose(result, [7.6590743])

    def test_multi_1hot(self):
        targets = tf.one_hot(tf.squeeze(MULTI_TARGETS, axis=-1), MULTI_LOGITS.shape[-1])
        result = crossentropy(
            y_true=targets, y_pred=MULTI_LOGITS, sample_weight=None, from_logits=True, force_binary=False,
            label_smoothing=0.)
        result = self.evaluate(result)
        self.assertAllClose(result, [5.34982])

    def test_multi_1hot_binary(self):
        targets = tf.one_hot(tf.squeeze(MULTI_TARGETS, axis=-1), MULTI_LOGITS.shape[-1])
        result = crossentropy(
            y_true=targets, y_pred=MULTI_LOGITS, sample_weight=None, from_logits=True, force_binary=True,
            label_smoothing=0.)
        result = self.evaluate(result)
        self.assertAllClose(result, [7.669404])

    def test_multi_1hot_smooth(self):
        targets = tf.one_hot(tf.squeeze(MULTI_TARGETS, axis=-1), MULTI_LOGITS.shape[-1])
        result = crossentropy(
            y_true=targets, y_pred=MULTI_LOGITS, sample_weight=None, from_logits=True, force_binary=False,
            label_smoothing=0.05)
        result = self.evaluate(result)
        self.assertAllClose(result, [5.34137])


@test_combinations.run_all_keras_modes
class TestIOU(test_combinations.TestCase):
    def test_zeros(self):
        logits = -10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_ones(self):
        logits = 10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_false(self):
        logits = -10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [1.] * 3, atol=6e-3)

    def test_true(self):
        logits = 10 * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [1.] * 3, atol=6e-3)

    def test_value(self):
        result = iou(y_true=BINARY_TARGETS, y_pred=BINARY_LOGITS, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.5122354, 0.5654068])

    def test_weight(self):
        result = iou(
            y_true=BINARY_TARGETS[:, :, :2], y_pred=BINARY_LOGITS[:, :, :2], sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.56775665, 0.263336])

        result = iou(y_true=BINARY_TARGETS, y_pred=BINARY_LOGITS, sample_weight=BINARY_WEIGHTS, from_logits=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.61162996, 0.29159677])

        result = iou(y_true=BINARY_TARGETS, y_pred=BINARY_LOGITS, sample_weight=BINARY_WEIGHTS * 2, from_logits=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.6362138, 0.30826524])

    def test_multi(self):
        result = iou(y_true=MULTI_TARGETS, y_pred=MULTI_LOGITS, sample_weight=None, from_logits=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.68037534])


@test_combinations.run_all_keras_modes
class TestDice(test_combinations.TestCase):
    def test_zeros(self):
        logits = -10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_ones(self):
        logits = 10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.] * 3, atol=6e-3)

    def test_false(self):
        logits = tf.ones((3, 16, 16, 1), 'float32') * (-10.)
        targets = tf.ones((3, 16, 16, 1), 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [1.] * 3, atol=6e-3)

    def test_true(self):
        logits = 10. * tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'int32')

        result = iou(y_true=targets, y_pred=logits, sample_weight=None, from_logits=True, dice=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [1.] * 3, atol=6e-3)

    def test_value(self):
        result = iou(y_true=BINARY_TARGETS, y_pred=BINARY_LOGITS, sample_weight=None, from_logits=True, dice=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.37031713, 0.43179172])

    def test_weight(self):
        result = iou(
            y_true=BINARY_TARGETS[:, :, :2], y_pred=BINARY_LOGITS[:, :, :2], sample_weight=None, from_logits=True,
            dice=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.44075716, 0.17269272])

        result = iou(
            y_true=BINARY_TARGETS, y_pred=BINARY_LOGITS, sample_weight=BINARY_WEIGHTS, from_logits=True, dice=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.46677598, 0.18477038])

        result = iou(
            y_true=BINARY_TARGETS, y_pred=BINARY_LOGITS, sample_weight=BINARY_WEIGHTS * 2, from_logits=True, dice=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.48097467, 0.19151434])

    def test_multi(self):
        result = iou(y_true=MULTI_TARGETS, y_pred=MULTI_LOGITS, sample_weight=None, from_logits=True, dice=True)
        result = self.evaluate(result)
        self.assertAllClose(result, [0.6068242])


BINARY_LOGITS = tf.constant([
    [[[0.4250706654827763], [7.219920928747051], [7.14131948950217], [-2.5576064452206024]],
     [[1.342442193620409], [0.20020616879804165], [-3.977300484664198], [6.280817910206608]],
     [[0.3206719246447576], [-3.0176225602425912], [2.902292891065069], [3.369106587128292]],
     [[-2.6576544216404563], [6.863726154333165], [4.581314280496405], [7.433728759092233]]],
    [[[-8.13888654097292], [8.311411218599392], [0.8372454481780323], [2.859455217953778]],
     [[2.0984725413538854], [-4.619268334888168], [8.708732477440673], [1.9102341271004541]],
     [[3.4914178176388266], [4.551627675234152], [-7.709902261544302], [3.3982255596983277]],
     [[0.9182162683255968], [3.0387004793287886], [2.1883984916630697], [-1.3921544038795197]]]], 'float32')
BINARY_TARGETS = tf.constant([
    [[[0], [0], [1], [0]], [[1], [0], [1], [1]], [[0], [1], [0], [1]], [[0], [1], [1], [1]]],
    [[[0], [1], [1], [0]], [[1], [0], [0], [1]], [[0], [1], [1], [0]], [[1], [1], [1], [1]]]], 'int32')
BINARY_WEIGHTS = tf.concat([tf.ones((2, 4, 2, 1)), tf.zeros((2, 4, 2, 1))], axis=2)

MULTI_LOGITS = tf.constant([
    [[[0.4250706654827763, -7.219920928747051, -1.14131948950217, 2.5576064452206024],
      [-1.342442193620409, 0.20020616879804165, -6.977300484664198, 6.280817910206608]],
     [[0.3206719246447576, 0.0176225602425912, -1.902292891065069, -3.369106587128292],
      [-2.6576544216404563, 1.863726154333165, 4.581314280496405, -7.433728759092233]],
     [[8.13888654097292, 1.311411218599392, 0.8372454481780323, -2.859455217953778],
      [-2.0984725413538854, -4.619268334888168, 8.708732477440673, 1.9102341271004541]],
     [[3.4914178176388266, -4.551627675234152, 7.709902261544302, 3.3982255596983277],
      [-0.9182162683255968, -7.0387004793287886, 2.1883984916630697, 1.3921544038795197]]]], 'float32')
MULTI_TARGETS = tf.constant([[[[1], [3]], [[3], [3]], [[1], [2]], [[2], [1]]]], 'int32')
MULTI_WEIGHTS = tf.concat([tf.ones((1, 4, 1, 1)), tf.zeros((1, 4, 1, 1))], axis=2)

if __name__ == '__main__':
    tf.test.main()
