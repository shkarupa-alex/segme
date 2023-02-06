import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.mixed_precision import policy as mixed_precision
from keras.utils import custom_object_scope
from keras.testing_infra import test_combinations, test_utils
from segme.common.pad import SymmetricPadding, with_divisible_pad


@test_combinations.run_all_keras_modes
class TestSymmetricPadding(test_combinations.TestCase):
    def setUp(self):
        super(TestSymmetricPadding, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSymmetricPadding, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            SymmetricPadding,
            kwargs={'padding': 1},
            input_shape=[2, 4, 5, 3],
            input_dtype='float32',
            expected_output_shape=[None, 6, 7, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            SymmetricPadding,
            kwargs={'padding': 1},
            input_shape=[2, 4, 5, 3],
            input_dtype='float16',
            expected_output_shape=[None, 6, 7, 3],
            expected_output_dtype='float16'
        )

    def test_error(self):
        with self.assertRaisesRegex(ValueError, 'Symmetric padding can lead to misbehavior'):
            SymmetricPadding(((0, 1), (1, 2)))


class OddConstrainedLayer(layers.Layer):
    def __init__(self, use_proj=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_proj = use_proj

    def build(self, input_shape):
        if self.use_proj:
            self.proj = layers.Conv2D(input_shape[-1] * 4, 3, padding='same')

        super().build(input_shape)

    def constraned_op(self, inputs, pad_size, pad_val):
        assert 3 == len(pad_size)
        assert 4 == len(pad_val)
        outputs = tf.nn.space_to_depth(inputs, 2)
        if self.use_proj:
            outputs = self.proj(outputs)
        outputs -= 1.
        outputs = tf.nn.depth_to_space(outputs, 2)

        return outputs

    def call(self, inputs, *args, **kwargs):
        outputs = with_divisible_pad(self.constraned_op, inputs, 2)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'use_proj': self.use_proj})

        return config


@test_combinations.run_all_keras_modes
class TestWithDivisiblePad(test_combinations.TestCase):
    def setUp(self):
        super(TestWithDivisiblePad, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestWithDivisiblePad, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        with custom_object_scope({'OddConstrainedLayer': OddConstrainedLayer}):
            test_utils.layer_test(
                OddConstrainedLayer,
                kwargs={'use_proj': True},
                input_shape=[2, 8, 10, 3],
                input_dtype='float32',
                expected_output_shape=[None, 8, 10, 3],
                expected_output_dtype='float32'
            )

    def test_value(self):
        inputs = np.arange(2 * 3 * 5 * 4).astype('float32').reshape([2, 3, 5, 4])

        result = OddConstrainedLayer(use_proj=False)(inputs)
        result = self.evaluate(result)
        self.assertAllClose(result, inputs - 1.)

    def test_grad(self):
        inputs = layers.Input(shape=(None, None, 3))
        outputs = OddConstrainedLayer(use_proj=True)(inputs)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile('adam', 'mse', jit_compile=True)
        model.fit(np.random.uniform(size=(16, 8, 10, 3)), np.random.uniform(size=(16, 8, 10, 3)))


if __name__ == '__main__':
    tf.test.main()
