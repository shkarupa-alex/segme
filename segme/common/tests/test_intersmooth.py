import tensorflow as tf
from keras import layers
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations, test_utils
from segme.common.intersmooth import SmoothInterpolation
from segme.policy import respol, resize
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestSmoothInterpolation(test_combinations.TestCase):
    def setUp(self):
        super(TestSmoothInterpolation, self).setUp()
        self.default_resize = respol.global_policy()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSmoothInterpolation, self).tearDown()
        respol.set_global_policy(self.default_resize)
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            SmoothInterpolation,
            kwargs={'scale': None},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float32']
        )
        test_utils.layer_test(
            SmoothInterpolation,
            kwargs={'scale': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 32, 32, 3],
            expected_output_dtype='float32'
        )

        with respol.policy_scope('inter_liif4'):
            layer_multi_io_test(
                SmoothInterpolation,
                kwargs={'scale': None},
                input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
                input_dtypes=['float32', 'float32'],
                expected_output_shapes=[(None, 24, 32, 10)],
                expected_output_dtypes=['float32']
            )
            test_utils.layer_test(
                SmoothInterpolation,
                kwargs={'scale': 2},
                input_shape=[2, 16, 16, 3],
                input_dtype='float32',
                expected_output_shape=[None, 32, 32, 3],
                expected_output_dtype='float32'
            )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            SmoothInterpolation,
            kwargs={'scale': None},
            input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
            input_dtypes=['float16', 'float16'],
            expected_output_shapes=[(None, 24, 32, 10)],
            expected_output_dtypes=['float16']
        )
        test_utils.layer_test(
            SmoothInterpolation,
            kwargs={'scale': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 32, 32, 3],
            expected_output_dtype='float16'
        )

        with respol.policy_scope('inter_liif4'):
            layer_multi_io_test(
                SmoothInterpolation,
                kwargs={'scale': None},
                input_shapes=[(2, 16, 16, 10), (2, 24, 32, 3)],
                input_dtypes=['float16', 'float16'],
                expected_output_shapes=[(None, 24, 32, 10)],
                expected_output_dtypes=['float16']
            )
            test_utils.layer_test(
                SmoothInterpolation,
                kwargs={'scale': 2},
                input_shape=[2, 16, 16, 3],
                input_dtype='float16',
                expected_output_shape=[None, 32, 32, 3],
                expected_output_dtype='float16'
            )

    def test_linear(self):
        res = SmoothInterpolation(2)
        res.build([None, None, None, 3])

        self.assertIsInstance(res.resize, resize.BilinearInterpolation)
        self.assertEqual(res.resize.scale, 2)

    def test_policy_scope_memorize(self):
        with respol.policy_scope('inter_liif4'):
            res = SmoothInterpolation(2)
        res.build([None, None, None, 3])

        self.assertIsInstance(res.resize, resize.LIIFInterpolation)
        self.assertEqual(res.resize.scale, 2)

        restored = SmoothInterpolation.from_config(res.get_config())
        restored.build([None, None, None, 3])
        self.assertIsInstance(restored.resize, resize.LIIFInterpolation)
        self.assertEqual(restored.resize.scale, 2)

    def test_policy_override_kwargs(self):
        with respol.policy_scope('inter_linear'):
            res = SmoothInterpolation(2, policy='inter_liif4')
        res.build([None, None, None, 3])

        restored = SmoothInterpolation.from_config(res.get_config())
        restored.build([None, None, None, 3])
        self.assertIsInstance(restored.resize, resize.LIIFInterpolation)


if __name__ == '__main__':
    tf.test.main()
