import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from ..dfeatalign import FeatureSelection, DeformableFeatureAlignment
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
class TestDeformableFeatureAlignment(test_combinations.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            DeformableFeatureAlignment,
            kwargs={'filters': 12, 'deformable_groups': 8},
            input_shapes=[(2, 8, 8, 12), (2, 16, 16, 5)],
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, 16, 16, 12)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
