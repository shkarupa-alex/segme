import tensorflow as tf
import unittest
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from segme.policy.resize import RESIZERS, LIIFInterpolation, GrIIFInterpolation, LGrIIFInterpolation
from segme.testing_utils import layer_multi_io_test


class TestResizeRegistry(unittest.TestCase):
    def test_filled(self):
        self.assertIn('inter_linear', RESIZERS)
        self.assertIn('inter_liif', RESIZERS)


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


@test_combinations.run_all_keras_modes
class TestGrIIFInterpolation(test_combinations.TestCase):
    def setUp(self):
        super(TestGrIIFInterpolation, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestGrIIFInterpolation, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            GrIIFInterpolation,
            kwargs={'scale': None, 'multi_scale': False, 'learn_positions': True, 'symmetric_pad': True},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float32']
        )
        test_utils.layer_test(
            GrIIFInterpolation,
            kwargs={'scale': 0.5, 'multi_scale': True, 'learn_positions': True, 'symmetric_pad': True},
            input_shape=(2, 16, 16, 10),
            input_dtype='float32',
            expected_output_shape=(None, 8, 8, 10),
            expected_output_dtype='float32'
        )
        layer_multi_io_test(
            GrIIFInterpolation,
            kwargs={'scale': None, 'multi_scale': False, 'learn_positions': False, 'symmetric_pad': True},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float32']
        )
        test_utils.layer_test(
            GrIIFInterpolation,
            kwargs={'scale': 0.5, 'multi_scale': False, 'learn_positions': True, 'symmetric_pad': False},
            input_shape=(2, 16, 16, 10),
            input_dtype='float32',
            expected_output_shape=(None, 8, 8, 10),
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            GrIIFInterpolation,
            kwargs={'scale': None, 'multi_scale': False, 'learn_positions': True, 'symmetric_pad': True},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float16', 'float16'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float16']
        )
        test_utils.layer_test(
            GrIIFInterpolation,
            kwargs={'scale': 2, 'multi_scale': True, 'learn_positions': True, 'symmetric_pad': True},
            input_shape=(2, 16, 16, 10),
            input_dtype='float16',
            expected_output_shape=(None, 32, 32, 10),
            expected_output_dtype='float16'
        )
        layer_multi_io_test(
            GrIIFInterpolation,
            kwargs={'scale': None, 'multi_scale': False, 'learn_positions': False, 'symmetric_pad': True},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float16', 'float16'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float16']
        )
        test_utils.layer_test(
            GrIIFInterpolation,
            kwargs={'scale': 2, 'multi_scale': False, 'learn_positions': True, 'symmetric_pad': False},
            input_shape=(2, 16, 16, 10),
            input_dtype='float16',
            expected_output_shape=(None, 32, 32, 10),
            expected_output_dtype='float16'
        )


@test_combinations.run_all_keras_modes
class TestLGrIIFInterpolation(test_combinations.TestCase):
    def setUp(self):
        super(TestLGrIIFInterpolation, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestLGrIIFInterpolation, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            LGrIIFInterpolation,
            kwargs={'scale': None, 'multi_scale': False, 'learn_positions': True, 'symmetric_pad': True},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float32']
        )
        test_utils.layer_test(
            LGrIIFInterpolation,
            kwargs={'scale': 0.5, 'multi_scale': True, 'learn_positions': True, 'symmetric_pad': True},
            input_shape=(2, 16, 16, 10),
            input_dtype='float32',
            expected_output_shape=(None, 8, 8, 10),
            expected_output_dtype='float32'
        )
        layer_multi_io_test(
            LGrIIFInterpolation,
            kwargs={'scale': None, 'multi_scale': False, 'learn_positions': False, 'symmetric_pad': True},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float32']
        )
        test_utils.layer_test(
            LGrIIFInterpolation,
            kwargs={'scale': 0.5, 'multi_scale': False, 'learn_positions': True, 'symmetric_pad': False},
            input_shape=(2, 16, 16, 10),
            input_dtype='float32',
            expected_output_shape=(None, 8, 8, 10),
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            LGrIIFInterpolation,
            kwargs={'scale': None, 'multi_scale': False, 'learn_positions': True, 'symmetric_pad': True},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float16', 'float16'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float16']
        )
        test_utils.layer_test(
            LGrIIFInterpolation,
            kwargs={'scale': 2, 'multi_scale': True, 'learn_positions': True, 'symmetric_pad': True},
            input_shape=(2, 16, 16, 10),
            input_dtype='float16',
            expected_output_shape=(None, 32, 32, 10),
            expected_output_dtype='float16'
        )
        layer_multi_io_test(
            LGrIIFInterpolation,
            kwargs={'scale': None, 'multi_scale': False, 'learn_positions': False, 'symmetric_pad': True},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float16', 'float16'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float16']
        )
        test_utils.layer_test(
            LGrIIFInterpolation,
            kwargs={'scale': 2, 'multi_scale': False, 'learn_positions': True, 'symmetric_pad': False},
            input_shape=(2, 16, 16, 10),
            input_dtype='float16',
            expected_output_shape=(None, 32, 32, 10),
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
