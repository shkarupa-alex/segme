import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from segme.model.sod.u2_net.rsu4 import RSU4


@test_combinations.run_all_keras_modes
class TestRSU4(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            RSU4,
            kwargs={'mid_features': 5, 'out_features': 4},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
