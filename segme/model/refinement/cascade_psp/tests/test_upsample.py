import tensorflow as tf
from tf_keras.src.testing_infra import test_combinations
from segme.model.refinement.cascade_psp.upsample import Upsample
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestUpsample(test_combinations.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Upsample,
            kwargs={'filters': 5},
            input_shapes=[(2, 4, 4, 3), (2, 16, 16, 3)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 16, 16, 5)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
