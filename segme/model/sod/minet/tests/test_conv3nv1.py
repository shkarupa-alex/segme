import tensorflow as tf
from keras.testing_infra import test_combinations
from segme.model.sod.minet.conv3nv1 import Conv3nV1
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestConv3nV1(test_combinations.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Conv3nV1,
            kwargs={'filters': 7},
            input_shapes=[(2, 16, 16, 2), (2, 8, 8, 4), (2, 4, 4, 2)],
            input_dtypes=['float32', 'float32', 'float32'],
            expected_output_shapes=[(None, 8, 8, 7)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
