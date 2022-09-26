import tensorflow as tf
from keras.testing_infra import test_combinations
from segme.model.segmentation.uper_net.head import Head
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestHead(test_combinations.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Head,
            kwargs={'classes': 2, 'dropout': 0.1},
            input_shapes=[(2, 64, 64, 4), (2, 128, 128, 3)],
            input_dtypes=['float32', 'uint8'],
            expected_output_shapes=[(None, 128, 128, 2)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
