import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations, test_utils
from segme.common.adppool import AdaptiveAveragePooling, AdaptiveMaxPooling


@test_combinations.run_all_keras_modes
class TestAdaptiveAveragePooling(test_combinations.TestCase):
    def setUp(self):
        super(TestAdaptiveAveragePooling, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestAdaptiveAveragePooling, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            AdaptiveAveragePooling,
            kwargs={'output_size': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 2, 2, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            AdaptiveAveragePooling,
            kwargs={'output_size': (4, 3)},
            input_shape=[2, 15, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 3, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            AdaptiveAveragePooling,
            kwargs={'output_size': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 2, 2, 3],
            expected_output_dtype='float16'
        )
        test_utils.layer_test(
            AdaptiveAveragePooling,
            kwargs={'output_size': (4, 3)},
            input_shape=[2, 15, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 4, 3, 3],
            expected_output_dtype='float16'
        )

    def test_value(self):
        shape = [2, 16, 16, 3]
        data = np.arange(0, np.prod(shape)).reshape(shape).astype('float32')

        result = test_utils.layer_test(
            AdaptiveAveragePooling,
            kwargs={'output_size': 1},
            input_data=data,
            expected_output_shape=[None, 1, 1, 3],
            expected_output_dtype='float32'
        ).astype('int32')
        self.assertListEqual(result.ravel().tolist(), [382, 383, 384, 1150, 1151, 1152])

        result = test_utils.layer_test(
            AdaptiveAveragePooling,
            kwargs={'output_size': 2},
            input_data=data,
            expected_output_shape=[None, 2, 2, 3],
            expected_output_dtype='float32'
        ).astype('int32')
        self.assertListEqual(result.ravel().tolist(), [
            178, 179, 180, 202, 203, 204, 562, 563, 564, 586, 587, 588, 946, 947, 948, 970, 971, 972, 1330, 1331, 1332,
            1354, 1355, 1356])

        result = test_utils.layer_test(
            AdaptiveAveragePooling,
            kwargs={'output_size': 3},
            input_data=data,
            expected_output_shape=[None, 3, 3, 3],
            expected_output_dtype='float32'
        ).astype('int32')
        self.assertListEqual(result.ravel().tolist(), [
            127, 128, 129, 142, 143, 144, 157, 158, 159, 367, 368, 369, 382, 383, 384, 397, 398, 399, 607, 608, 609,
            622, 623, 624, 637, 638, 639, 895, 896, 897, 910, 911, 912, 925, 926, 927, 1135, 1136, 1137, 1150, 1151,
            1152, 1165, 1166, 1167, 1375, 1376, 1377, 1390, 1391, 1392, 1405, 1406, 1407])

    def test_placeholder(self):
        shape = [2, 16, 16, 3]
        data = np.arange(0, np.prod(shape)).reshape(shape).astype('float32')
        target = np.random.uniform(size=(2, 3, 3, 3))
        dataset = tf.data.Dataset.from_tensor_slices((data, target)).batch(2)

        inputs = layers.Input([None, None, 3], dtype='float32')
        outputs = AdaptiveAveragePooling(3)(inputs)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile('adam', 'mse', run_eagerly=test_utils.should_run_eagerly())
        model.fit(dataset)


@test_combinations.run_all_keras_modes
class TestAdaptiveMaxPooling(test_combinations.TestCase):
    def setUp(self):
        super(TestAdaptiveMaxPooling, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestAdaptiveMaxPooling, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            AdaptiveMaxPooling,
            kwargs={'output_size': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 2, 2, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            AdaptiveMaxPooling,
            kwargs={'output_size': (4, 3)},
            input_shape=[2, 15, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 3, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            AdaptiveMaxPooling,
            kwargs={'output_size': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 2, 2, 3],
            expected_output_dtype='float16'
        )
        test_utils.layer_test(
            AdaptiveMaxPooling,
            kwargs={'output_size': (4, 3)},
            input_shape=[2, 15, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 4, 3, 3],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
