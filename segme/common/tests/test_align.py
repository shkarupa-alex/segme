import numpy as np
import tensorflow as tf
from keras import layers
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations
from segme.common.align import Align
from segme.policy import alpol
from segme.policy.align import align
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestAlign(test_combinations.TestCase):
    def setUp(self):
        super(TestAlign, self).setUp()
        self.default_Alignnormact = alpol.global_policy()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestAlign, self).tearDown()
        alpol.set_global_policy(self.default_Alignnormact)
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            Align,
            kwargs={'filters': 4},
            input_shapes=[(2, 16, 16, 3), (2, 8, 8, 6)],
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, 16, 16, 4)],
            expected_output_dtypes=['float32']
        )

        with alpol.policy_scope('deconv4'):
            layer_multi_io_test(
                Align,
                kwargs={'filters': 4},
                input_shapes=[(2, 16, 16, 3), (2, 8, 8, 6)],
                input_dtypes=['float32'] * 2,
                expected_output_shapes=[(None, 16, 16, 4)],
                expected_output_dtypes=['float32']
            )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            Align,
            kwargs={'filters': 4},
            input_shapes=[(2, 16, 16, 3), (2, 8, 8, 6)],
            input_dtypes=['float16'] * 2,
            expected_output_shapes=[(None, 16, 16, 4)],
            expected_output_dtypes=['float16']
        )

        with alpol.policy_scope('deconv4'):
            layer_multi_io_test(
                Align,
                kwargs={'filters': 4},
                input_shapes=[(2, 16, 16, 3), (2, 8, 8, 6)],
                input_dtypes=['float16'] * 2,
                expected_output_shapes=[(None, 16, 16, 4)],
                expected_output_dtypes=['float16']
            )

    def test_linear(self):
        aligninst = Align(4)
        aligninst.build([[None, None, None, 3], [None, None, None, 3]])

        self.assertIsInstance(aligninst, align.BilinearFeatureAlignment)

    def test_policy_scope(self):
        with alpol.policy_scope('sapa'):
            aligninst = Align(4)
        aligninst.build([[None, None, None, 3], [None, None, None, 3]])

        self.assertIsInstance(aligninst, align.SapaFeatureAlignment)

    def test_shape(self):
        fine = np.zeros((2, 16, 16, 16), dtype='float32')
        coarse = np.zeros((2, 8, 8, 32), dtype='float32')

        for method in align.ALIGNERS.keys():
            with alpol.policy_scope(method):
                result = Align(4)([fine, coarse])
                result = self.evaluate(result)
                self.assertTupleEqual(result.shape, (2, 16, 16, 4))


if __name__ == '__main__':
    tf.test.main()
