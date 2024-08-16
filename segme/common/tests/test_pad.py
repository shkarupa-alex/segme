import numpy as np
import tensorflow as tf
from keras.src import layers
from keras.src import models
from keras.src import testing

from segme.common.pad import SymmetricPadding
from segme.common.pad import with_divisible_pad


class TestSymmetricPadding(testing.TestCase):
    

    def test_layer(self):
        self.run_layer_test(
            SymmetricPadding,
            init_kwargs={"padding": 1},
            input_shape=(2, 4, 5, 3),
            input_dtype="float32",
            expected_output_shape=(2, 6, 7, 3),
            expected_output_dtype="float32",
        )

    def test_error(self):
        with self.assertRaisesRegex(
            ValueError, "Symmetric padding can lead to misbehavior"
        ):
            SymmetricPadding(((0, 1), (1, 2)))


class OddConstrainedLayer(layers.Layer):
    def __init__(self, use_proj=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_proj = use_proj

    def build(self, input_shape):
        if self.use_proj:
            self.proj = layers.Conv2D(input_shape[-1] * 4, 3, padding="same")
            self.proj.build(input_shape[:-1] + (input_shape[-1] * 4,))

        super().build(input_shape)

    def constraned_op(self, inputs, pad_size, pad_val):
        assert 2 == len(pad_size)
        assert 4 == len(pad_val)
        outputs = tf.nn.space_to_depth(inputs, 2)
        if self.use_proj:
            outputs = self.proj(outputs)
        outputs -= 1.0
        outputs = tf.nn.depth_to_space(outputs, 2)

        return outputs

    def call(self, inputs, *args, **kwargs):
        outputs = with_divisible_pad(self.constraned_op, inputs, 2)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"use_proj": self.use_proj})

        return config


class TestWithDivisiblePad(testing.TestCase):
    

    def test_layer(self):
        self.run_layer_test(
            OddConstrainedLayer,
            init_kwargs={"use_proj": True},
            input_shape=(2, 8, 10, 3),
            input_dtype="float32",
            expected_output_shape=(2, 8, 10, 3),
            expected_output_dtype="float32",
            custom_objects={"OddConstrainedLayer": OddConstrainedLayer}

        )

    def test_value(self):
        inputs = (
            np.arange(2 * 3 * 5 * 4).astype("float32").reshape([2, 3, 5, 4])
        )

        result = OddConstrainedLayer(use_proj=False)(inputs)
        self.assertAllClose(result, inputs - 1.0)

    def test_grad(self):
        inputs = layers.Input(shape=(None, None, 3))
        outputs = OddConstrainedLayer(use_proj=True)(inputs)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile("adam", "mse", jit_compile=True)
        model.fit(
            np.random.uniform(size=(16, 8, 10, 3)),
            np.random.uniform(size=(16, 8, 10, 3)),
        )
