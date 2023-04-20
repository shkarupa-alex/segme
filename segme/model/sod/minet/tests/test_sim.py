import tensorflow as tf
from keras.src.testing_infra import test_combinations, test_utils
from segme.model.sod.minet.sim import SIM


@test_combinations.run_all_keras_modes
class TestSim(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            SIM,
            kwargs={'filters': 4},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SIM,
            kwargs={'filters': 10},
            input_shape=[2, 17, 17, 3],
            input_dtype='float32',
            expected_output_shape=[None, 17, 17, 3],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
