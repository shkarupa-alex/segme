import numpy as np
import tensorflow as tf
from keras.src import backend
from keras.src import layers
from keras.src import testing

from segme.common.drop import DropBlock
from segme.common.drop import DropPath
from segme.common.drop import RestorePath
from segme.common.drop import SlicePath


class TestDropPath(testing.TestCase):

    def test_layer(self):
        self.run_layer_test(
            DropPath,
            init_kwargs={"rate": 0.0},
            input_shape=(2, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 3),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            DropPath,
            init_kwargs={"rate": 0.2},
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 3),
            expected_output_dtype="float32",
        )

    def test_val(self):
        inputs = tf.ones([20, 4], "float32")
        result = DropPath(0.2, seed=1)(inputs, training=True)
        result = backend.convert_to_numpy(result)
        self.assertSetEqual(set(result.ravel()), {0.0, 1.25})
        self.assertEqual((result == 0.0).all(axis=-1).mean(), 0.2)


class SliceRestorePath(layers.Layer):
    def __init__(self, rate, seed=None, **kwargs):
        super().__init__(**kwargs)

        self.rate = rate
        self.seed = seed

    def build(self, input_shape):
        self.slice = SlicePath(self.rate, self.seed)
        self.restore = RestorePath(self.rate, self.seed)

    def call(self, inputs, training=None, *args, **kwargs):
        outputs, masks = self.slice(inputs, training=training)
        outputs = self.restore([outputs, masks], training=training)

        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = self.slice.compute_output_shape(input_shape)
        output_shape = self.restore.compute_output_shape(output_shape)

        return output_shape

    def compute_output_signature(self, input_signature):
        output_signature = self.slice.compute_output_signature(input_signature)
        output_signature = self.restore.compute_output_signature(
            output_signature
        )

        return output_signature

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate, "seed": self.seed})

        return config


class TestSliceRestorePath(testing.TestCase):

    def test_layer(self):
        self.run_layer_test(
            SliceRestorePath,
            init_kwargs={"rate": 0.0},
            input_shape=(20, 4, 3),
            input_dtype="float32",
            expected_output_shape=(2, 4, 3),
            expected_output_dtype="float32",
            custom_objects={"SliceRestorePath": SliceRestorePath},
        )
        self.run_layer_test(
            SliceRestorePath,
            init_kwargs={"rate": 0.2},
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 3),
            expected_output_dtype="float32",
            custom_objects={"SliceRestorePath": SliceRestorePath},
        )

    def test_val(self):
        inputs = tf.ones([20, 4], "float32")
        result = SliceRestorePath(0.2, seed=1)(inputs, training=True)
        result = backend.convert_to_numpy(result)
        self.assertSetEqual(set(result.ravel()), {0.0, 1.25})
        self.assertEqual((result == 0.0).all(axis=-1).mean(), 0.2)


class TestDropBlock(testing.TestCase):

    def test_layer(self):
        self.run_layer_test(
            DropBlock,
            init_kwargs={"rate": 0.0, "size": 2},
            input_shape=(2, 16, 3, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 3, 3),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            DropBlock,
            init_kwargs={"rate": 0.2, "size": 1},
            input_shape=(2, 16, 16, 4),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 4),
            expected_output_dtype="float32",
        )

    def test_mean(self):
        inputs = np.random.uniform(size=[4, 32, 32, 3]).astype("float32")

        outputs = DropBlock(0.2, 3)(inputs, training=True)
        outputs = backend.convert_to_numpy(outputs)

        self.assertNotAllClose(inputs, outputs)
        self.assertAllClose(
            inputs.mean(axis=(1, 2)), outputs.mean(axis=(1, 2)), atol=5e-2
        )
