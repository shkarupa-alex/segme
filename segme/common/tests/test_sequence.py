import numpy as np
import tensorflow as tf
from keras.src import layers, backend
from keras.src import testing
from keras.src import utils

from segme.common.sequence import Sequence


class Split2(layers.Layer):
    def call(self, inputs, *args, **kwargs):
        return tuple(tf.split(inputs, 2, axis=-1))

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask, mask

    def compute_output_shape(self, input_shape):
        return (input_shape[:-1] + (input_shape[-1] //2 ,),)*2



class TestSequence(testing.TestCase):
    

    def test_layer(self):
        self.run_layer_test(
            Sequence,
            init_kwargs={
                "items": [layers.BatchNormalization(), layers.ReLU()]
            },
            input_shape=(2, 16, 16, 10),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 10),
            expected_output_dtype="float32",

        )
        self.run_layer_test(
            Sequence,
            init_kwargs={"items": [layers.ReLU(), Split2()]},
            input_shape=(2, 16, 16, 10),
            input_dtype="float32",
            expected_output_shape=((2, 16, 16, 5), (2, 16, 16, 5)),
            expected_output_dtype=("float32",) * 2,
            custom_objects={"Split2": Split2}

        )
        self.run_layer_test(
            Sequence,
            init_kwargs={"items": [layers.ReLU(), Split2(), layers.Add()]},
            input_shape=(2, 16, 16, 10),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 5),
            expected_output_dtype="float32",
            custom_objects={"Split2": Split2}

        )
        self.run_layer_test(
            Sequence,
            init_kwargs={
                "items": [
                    layers.BatchNormalization(),
                    layers.ReLU(dtype="float32"),
                ]
            },
            input_shape=(2, 16, 16, 10),
            input_dtype="float16",
            expected_output_shape=(2, 16, 16, 10),
            expected_output_dtype="float32",
            run_mixed_precision_check=False
        )

    def test_mask_produce(self):
        inputs = np.array([
            [0, 1, 2],
            [2, 1, 0]
        ]).astype('int32')

        outputs = Sequence([
            layers.Embedding(3, 4, mask_zero=True),
            layers.Dense(6),
            layers.ReLU(),
            Split2(),
            layers.Minimum()
        ])(inputs)
        self.assertTrue(hasattr(outputs, '_keras_mask'))

        mask = backend.convert_to_numpy(outputs._keras_mask)
        self.assertTrue((mask == np.array([[False,  True,  True], [ True,  True, False]])).all())


    def test_mask_pass(self):
        inputs = np.random.uniform(size=[2, 3, 4]).astype('float32')
        masks = np.array([[False,  True,  True], [ True,  True, False]])

        outputs = Sequence([
            layers.Dense(6),
            layers.ReLU(),
            Split2(),
            layers.Minimum()
        ])(inputs, mask=masks)
        self.assertTrue(hasattr(outputs, '_keras_mask'))

        mask = backend.convert_to_numpy(outputs._keras_mask)
        self.assertTrue((mask == masks).all())
