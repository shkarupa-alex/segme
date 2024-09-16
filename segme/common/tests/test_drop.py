import numpy as np
from keras.src import backend
from keras.src import ops
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
        inputs = ops.ones([20, 4], "float32")
        result = DropPath(0.2, seed=1)(inputs, training=True)
        result = backend.convert_to_numpy(result)
        self.assertSetEqual(set(result.ravel()), {0.0, 1.25})
        self.assertEqual((result == 0.0).all(axis=-1).mean(), 0.2)


class TestSliceRestorePath(testing.TestCase):
    def test_layer_slice(self):
        self.run_layer_test(
            SlicePath,
            init_kwargs={"rate": 0.0},
            input_shape=(2, 3),
            input_dtype="float32",
            expected_output_shape=((2, 3), (2,)),
            expected_output_dtype=("float32", "int32"),
        )
        self.run_layer_test(
            SlicePath,
            init_kwargs={"rate": 0.2, "seed": 1},
            input_shape=(60, 2, 3),
            input_dtype="float32",
            call_kwargs={"training": True},
            expected_output_shape=((48, 2, 3), (60,)),
            expected_output_dtype=("float32", "int32"),
        )

    def test_layer_restore(self):
        self.run_layer_test(
            RestorePath,
            init_kwargs={},
            input_shape=((2, 3), (2,)),
            input_dtype=("float32", "int32"),
            expected_output_shape=(2, 3),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            RestorePath,
            init_kwargs={},
            input_shape=((48, 2, 3), (60,)),
            input_dtype=("float32", "int32"),
            call_kwargs={"training": True},
            expected_output_shape=(60, 2, 3),
            expected_output_dtype="float32",
        )

    def test_val(self):
        inputs = np.ones([60, 4], "float32") * np.arange(1, 61)[:, None]
        result = SlicePath(0.2, seed=1)(inputs, training=True)
        result = RestorePath()(result, training=True)
        result = backend.convert_to_numpy(result)
        self.assertEqual((result == 0.0).all(axis=-1).mean(), 0.2)
        self.assertTrue(
            (np.where(result == 0.0, inputs, result * 0.8) == inputs).all()
        )


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
            init_kwargs={"rate": 0.2, "size": 20},
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
