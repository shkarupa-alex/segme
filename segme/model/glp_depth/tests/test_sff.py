import tensorflow as tf
from keras import keras_parameterized
from ..sff import SelectiveFeatureFusion
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestSelectiveFeatureFusion(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            SelectiveFeatureFusion,
            kwargs={'standardized': False},
            input_shapes=[(2, 32, 32, 32), (2, 32, 32, 32)],
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, 32, 32, 32)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            SelectiveFeatureFusion,
            kwargs={'standardized': True},
            input_shapes=[(2, 32, 32, 64), (2, 32, 32, 64)],
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, 32, 32, 64)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
