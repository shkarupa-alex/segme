import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from ..featalign import FeatureSelection, FeatureAlignment
from ...testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestFeatureSelection(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            FeatureSelection,
            kwargs={'filters': 4},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )


@test_combinations.run_all_keras_modes
class TestFeatureAlignment(test_combinations.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            FeatureAlignment,
            kwargs={'filters': 12, 'deformable_groups': 8},
            input_shapes=[(2, 16, 16, 12), (2, 16, 16, 5)],
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, 16, 16, 12)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
