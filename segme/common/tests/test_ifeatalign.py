import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from tensorflow.python.framework import test_util
from ..ifeatalign import ImplicitFeatureAlignment, SpatialEncoding
from ...testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestSpatialEncoding(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            SpatialEncoding,
            kwargs={'units': 24, 'sigma': 6},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 28],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SpatialEncoding,
            kwargs={'units': 24, 'sigma': 4},
            input_shape=[2, 16, 16, 1],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 25],
            expected_output_dtype='float32'
        )


@test_combinations.run_all_keras_modes
class TestImplicitFeatureAlignment(test_combinations.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            ImplicitFeatureAlignment,
            kwargs={'filters': 12},
            input_shapes=[(2, 16, 16, 2), (2, 8, 8, 5), (2, 4, 4, 10)],
            input_dtypes=['float32'] * 3,
            expected_output_shapes=[(None, 16, 16, 12)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
