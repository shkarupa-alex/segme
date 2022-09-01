import tensorflow as tf
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations, test_utils
from segme.common.align.deform import DeformableFeatureAlignment, FeatureSelection
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestDeformableFeatureAlignment(test_combinations.TestCase):
    def setUp(self):
        super(TestDeformableFeatureAlignment, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestDeformableFeatureAlignment, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            DeformableFeatureAlignment,
            kwargs={'filters': 8, 'deformable_groups': 8},
            input_shapes=[(2, 16, 16, 6), (2, 8, 8, 12)],
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, 16, 16, 8)],
            expected_output_dtypes=['float32']
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            DeformableFeatureAlignment,
            kwargs={'filters': 8, 'deformable_groups': 8},
            input_shapes=[(2, 16, 16, 6), (2, 8, 8, 12)],
            input_dtypes=['float16'] * 2,
            expected_output_shapes=[(None, 16, 16, 8)],
            expected_output_dtypes=['float16']
        )


@test_combinations.run_all_keras_modes
class TestFeatureSelection(test_combinations.TestCase):
    def setUp(self):
        super(TestFeatureSelection, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestFeatureSelection, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            FeatureSelection,
            kwargs={'filters': 4},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            FeatureSelection,
            kwargs={'filters': 4},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
