import numpy as np
import tensorflow as tf
import unittest
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from segme.policy.resize import RESIZERS, BilinearInterpolation, LIIFInterpolation
from segme.testing_utils import layer_multi_io_test


class TestNormalizationsRegistry(unittest.TestCase):
    def test_filled(self):
        self.assertIn('inter_linear', RESIZERS)
        self.assertIn('inter_liif4', RESIZERS)


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


@test_combinations.run_all_keras_modes
class TestLIIFInterpolation(test_combinations.TestCase):
    def setUp(self):
        super(TestLIIFInterpolation, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestLIIFInterpolation, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            LIIFInterpolation,
            kwargs={
                'scale': None, 'feat_unfold': False, 'local_ensemble': False, 'learn_positions': False,
                'symmetric_pad': False},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float32']
        )
        test_utils.layer_test(
            LIIFInterpolation,
            kwargs={'scale': 2, 'feat_unfold': False, 'local_ensemble': False, 'learn_positions': False,
                    'symmetric_pad': False},
            input_shape=(2, 16, 16, 10),
            input_dtype='float32',
            expected_output_shape=(None, 32, 32, 10),
            expected_output_dtype='float32'
        )
        layer_multi_io_test(
            LIIFInterpolation,
            kwargs={
                'scale': None, 'feat_unfold': 3, 'local_ensemble': False, 'learn_positions': False,
                'symmetric_pad': False},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float32']
        )
        test_utils.layer_test(
            LIIFInterpolation,
            kwargs={'scale': 2, 'feat_unfold': False, 'local_ensemble': True, 'learn_positions': False,
                    'symmetric_pad': False},
            input_shape=(2, 16, 16, 10),
            input_dtype='float32',
            expected_output_shape=(None, 32, 32, 10),
            expected_output_dtype='float32'
        )
        layer_multi_io_test(
            LIIFInterpolation,
            kwargs={
                'scale': None, 'feat_unfold': False, 'local_ensemble': False, 'learn_positions': True,
                'symmetric_pad': False},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float32']
        )
        test_utils.layer_test(
            LIIFInterpolation,
            kwargs={'scale': 2, 'feat_unfold': False, 'local_ensemble': False, 'learn_positions': False,
                    'symmetric_pad': True},
            input_shape=(2, 16, 16, 10),
            input_dtype='float32',
            expected_output_shape=(None, 32, 32, 10),
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            LIIFInterpolation,
            kwargs={
                'scale': None, 'feat_unfold': False, 'local_ensemble': False, 'learn_positions': False,
                'symmetric_pad': False},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float16', 'float16'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float16']
        )
        test_utils.layer_test(
            LIIFInterpolation,
            kwargs={
                'scale': 0.5, 'feat_unfold': False, 'local_ensemble': False, 'learn_positions': False,
                'symmetric_pad': False},
            input_shape=(2, 16, 16, 10),
            input_dtype='float16',
            expected_output_shape=(None, 8, 8, 10),
            expected_output_dtype='float16'
        )
        layer_multi_io_test(
            LIIFInterpolation,
            kwargs={
                'scale': None, 'feat_unfold': 3, 'local_ensemble': False, 'learn_positions': False,
                'symmetric_pad': False},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float16', 'float16'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float16']
        )
        test_utils.layer_test(
            LIIFInterpolation,
            kwargs={
                'scale': 0.5, 'feat_unfold': False, 'local_ensemble': True, 'learn_positions': False,
                'symmetric_pad': False},
            input_shape=(2, 16, 16, 10),
            input_dtype='float16',
            expected_output_shape=(None, 8, 8, 10),
            expected_output_dtype='float16'
        )
        layer_multi_io_test(
            LIIFInterpolation,
            kwargs={
                'scale': None, 'feat_unfold': False, 'local_ensemble': False, 'learn_positions': True,
                'symmetric_pad': False},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float16', 'float16'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float16']
        )
        test_utils.layer_test(
            LIIFInterpolation,
            kwargs={
                'scale': 0.5, 'feat_unfold': False, 'local_ensemble': False, 'learn_positions': False,
                'symmetric_pad': True},
            input_shape=(2, 16, 16, 10),
            input_dtype='float16',
            expected_output_shape=(None, 8, 8, 10),
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
