import tensorflow as tf
from tf_keras import mixed_precision
from tf_keras.src.testing_infra import test_combinations, test_utils
from segme.common.resize import NearestInterpolation, BilinearInterpolation
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestNearestInterpolation(test_combinations.TestCase):
    def setUp(self):
        super(TestNearestInterpolation, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestNearestInterpolation, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            NearestInterpolation,
            kwargs={'scale': None},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float32']
        )
        test_utils.layer_test(
            NearestInterpolation,
            kwargs={'scale': 2},
            input_shape=(2, 16, 16, 10),
            input_dtype='float32',
            expected_output_shape=(None, 32, 32, 10),
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            NearestInterpolation,
            kwargs={'scale': None},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float16', 'float16'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float16']
        )
        test_utils.layer_test(
            NearestInterpolation,
            kwargs={'scale': 0.5},
            input_shape=(2, 16, 16, 10),
            input_dtype='float16',
            expected_output_shape=(None, 8, 8, 10),
            expected_output_dtype='float16'
        )

    def test_tile(self):
        inputs = tf.random.uniform([3, 1, 1, 5])

        expected = tf.image.resize(inputs, [2, 7], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        expected = self.evaluate(expected)

        result = NearestInterpolation()([inputs, tf.zeros([1, 2, 7, 5])])
        result = self.evaluate(result)

        self.assertAllClose(expected, result)


@test_combinations.run_all_keras_modes
class TestBilinearInterpolation(test_combinations.TestCase):
    def setUp(self):
        super(TestBilinearInterpolation, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestBilinearInterpolation, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            BilinearInterpolation,
            kwargs={'scale': None},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float32']
        )
        test_utils.layer_test(
            BilinearInterpolation,
            kwargs={'scale': 2},
            input_shape=(2, 16, 16, 10),
            input_dtype='float32',
            expected_output_shape=(None, 32, 32, 10),
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            BilinearInterpolation,
            kwargs={'scale': None},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float16', 'float16'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float16']
        )
        test_utils.layer_test(
            BilinearInterpolation,
            kwargs={'scale': 0.5},
            input_shape=(2, 16, 16, 10),
            input_dtype='float16',
            expected_output_shape=(None, 8, 8, 10),
            expected_output_dtype='float16'
        )
        test_utils.layer_test(
            BilinearInterpolation,
            kwargs={'scale': 0.5, 'dtype': 'float32'},
            input_shape=(2, 16, 16, 10),
            input_dtype='float16',
            expected_output_shape=(None, 8, 8, 10),
            expected_output_dtype='float32'
        )

    # Current tf.image.resize implementation fully mimics cv2.imresize
    # def test_corners(self):
    #     target = tf.reshape(tf.range(9, dtype=tf.float32), [1, 3, 3, 1])
    #     sample = tf.zeros([1, 10, 9, 1], dtype=tf.float32)
    #     result = resize_by_sample([target, sample])
    #     result = self.evaluate(result)
    #
    #     # See https://github.com/tensorflow/tensorflow/issues/6720#issuecomment-644111750
    #     expected = np.array([
    #         [0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2.],
    #         [0.667, 0.917, 1.167, 1.417, 1.667, 1.917, 2.167, 2.417, 2.667],
    #         [1.333, 1.583, 1.833, 2.083, 2.333, 2.583, 2.833, 3.083, 3.333],
    #         [2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4.],
    #         [2.667, 2.917, 3.167, 3.417, 3.667, 3.917, 4.167, 4.417, 4.667],
    #         [3.333, 3.583, 3.833, 4.083, 4.333, 4.583, 4.833, 5.083, 5.333],
    #         [4., 4.25, 4.5, 4.75, 5., 5.25, 5.5, 5.75, 6.],
    #         [4.667, 4.917, 5.167, 5.417, 5.667, 5.917, 6.167, 6.417, 6.667],
    #         [5.333, 5.583, 5.833, 6.083, 6.333, 6.583, 6.833, 7.083, 7.333],
    #         [6., 6.25, 6.5, 6.75, 7., 7.25, 7.5, 7.75, 8.]
    #     ]).reshape([1, 10, 9, 1])
    #
    #     self.assertAllClose(result, expected, 0.002)


if __name__ == '__main__':
    tf.test.main()
