import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from segme.model.deeplab_v3_plus.base import DeepLabV3PlusBase


@test_combinations.run_all_keras_modes
class TestDeepLabV3PlusBase(test_combinations.TestCase):
    def setUp(self):
        super(TestDeepLabV3PlusBase, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestDeepLabV3PlusBase, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            DeepLabV3PlusBase,
            kwargs={
                'classes': 4, 'aspp_filters': 8, 'aspp_stride': 32, 'low_filters': 16, 'decoder_filters': 5},
            input_shape=(2, 224, 224, 3),
            input_dtype='uint8',
            expected_output_shape=(None, 56, 56, 4),
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            DeepLabV3PlusBase,
            kwargs={
                'classes': 1, 'aspp_filters': 8, 'aspp_stride': 32, 'low_filters': 16, 'decoder_filters': 4},
            input_shape=(2, 224, 224, 3),
            input_dtype='uint8',
            expected_output_shape=(None, 56, 56, 1),
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
