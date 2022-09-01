import tensorflow as tf
from keras import layers
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from keras.utils.generic_utils import custom_object_scope
from segme.common.hmsattn import HierarchicalMultiScaleAttention


class LogitsWithGuidance(layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(4, 3, strides=2, padding='same')
        self.conv2 = layers.Conv2D(2, 3, strides=2, padding='same')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        features = self.conv1(inputs)
        outputs = self.conv2(features)

        return outputs, features


@test_combinations.run_all_keras_modes
class TestHierarchicalMultiScaleAttention(test_combinations.TestCase):
    def setUp(self):
        super(TestHierarchicalMultiScaleAttention, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestHierarchicalMultiScaleAttention, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        with custom_object_scope({'LogitsWithGuidance': LogitsWithGuidance}):
            test_utils.layer_test(
                HierarchicalMultiScaleAttention,
                kwargs={'layer': LogitsWithGuidance(), 'scales': ((0.5,), (0.25, 0.5, 2.0)),
                        'filters': 256, 'dropout': 0.},
                input_shape=[2, 128, 128, 3],
                input_dtype='float32',
                expected_output_shape=[None, 128, 128, 2],
                expected_output_dtype='float32'
            )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        with custom_object_scope({'LogitsWithGuidance': LogitsWithGuidance}):
            test_utils.layer_test(
                HierarchicalMultiScaleAttention,
                kwargs={'layer': LogitsWithGuidance(), 'scales': ((0.5,), (0.5, 2.0)),
                        'filters': 256, 'dropout': 0.},
                input_shape=[2, 128, 128, 3],
                input_dtype='float16',
                expected_output_shape=[None, 128, 128, 2],
                expected_output_dtype='float16'
            )


if __name__ == '__main__':
    tf.test.main()
