import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from ..decoder import Decoder, Bottleneck, SqueezeExcitation
from ....testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestDecoder(test_combinations.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Decoder,
            kwargs={},
            input_shapes=[
                (2, 256, 256, 6), (2, 128, 128, 8), (2, 64, 64, 16), (2, 32, 32, 32), (2, 16, 16, 64), (2, 8, 8, 128)],
            input_dtypes=['float32'] * 6,
            expected_output_shapes=[(None, 256, 256, 6)],
            expected_output_dtypes=['float32']
        )

@test_combinations.run_all_keras_modes
class TestBottleneck(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            Bottleneck,
            kwargs={
                'filters': 2, 'strides': 1, 'use_projection': True, 'bn_momentum': 0.99, 'bn_epsilon': 1e-5,
                'activation': 'leaky_relu'},
            input_shape=(2, 16, 16, 8),
            input_dtype='float32',
            expected_output_shape=(None, 16, 16, 2),
        )
        test_utils.layer_test(
            Bottleneck,
            kwargs={
                'filters': 2, 'strides': 1, 'use_projection': False, 'bn_momentum': 0.99, 'bn_epsilon': 1e-5,
                'activation': 'leaky_relu'},
            input_shape=(2, 16, 16, 2),
            input_dtype='float32',
            expected_output_shape=(None, 16, 16, 2),
        )

@test_combinations.run_all_keras_modes
class TestSqueezeExcitation(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            SqueezeExcitation,
            kwargs={'ratio': 1},
            input_shape=(2, 16, 16, 3),
            input_dtype='float32',
            expected_output_shape=(None, 16, 16, 3),
        )
        test_utils.layer_test(
            SqueezeExcitation,
            kwargs={'ratio': 4},
            input_shape=(2, 16, 16, 8),
            input_dtype='float32',
            expected_output_shape=(None, 16, 16, 8),
        )

if __name__ == '__main__':
    tf.test.main()
