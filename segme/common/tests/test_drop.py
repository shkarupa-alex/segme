import numpy as np
import tensorflow as tf
from keras import mixed_precision, layers, utils
from keras.src.testing_infra import test_combinations, test_utils
from segme.common.drop import DropPath, SlicePath, RestorePath, DropBlock


@test_combinations.run_all_keras_modes
class TestDropPath(test_combinations.TestCase):
    def setUp(self):
        super(TestDropPath, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestDropPath, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            DropPath,
            kwargs={'rate': 0.},
            input_shape=[2, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            DropPath,
            kwargs={'rate': 0.2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            DropPath,
            kwargs={'rate': 0.1},
            input_shape=[2, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 3],
            expected_output_dtype='float16'
        )


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
        output_signature = self.restore.compute_output_signature(output_signature)

        return output_signature

    def get_config(self):
        config = super().get_config()
        config.update({'rate': self.rate, 'seed': self.seed})

        return config


@test_combinations.run_all_keras_modes
class TestSliceRestorePath(test_combinations.TestCase):
    def setUp(self):
        super(TestSliceRestorePath, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSliceRestorePath, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        with utils.custom_object_scope({'SliceRestorePath': SliceRestorePath}):
            test_utils.layer_test(
                SliceRestorePath,
                kwargs={'rate': 0.},
                input_shape=[20, 4, 3],
                input_dtype='float32',
                expected_output_shape=[None, 4, 3],
                expected_output_dtype='float32'
            )
            test_utils.layer_test(
                SliceRestorePath,
                kwargs={'rate': 0.2},
                input_shape=[2, 16, 16, 3],
                input_dtype='float32',
                expected_output_shape=[None, 16, 16, 3],
                expected_output_dtype='float32'
            )

    def test_val(self):
        with utils.custom_object_scope({'SliceRestorePath': SliceRestorePath}):
            inputs = tf.ones([20, 4], 'float32')
            result = SliceRestorePath(0.2)(inputs, training=True)
            result = self.evaluate(result)
            self.assertEqual(result.max(), 1.25)

    def test_fp16(self):
        with utils.custom_object_scope({'SliceRestorePath': SliceRestorePath}):
            mixed_precision.set_global_policy('mixed_float16')
            test_utils.layer_test(
                SliceRestorePath,
                kwargs={'rate': 0.1},
                input_shape=[2, 16, 3],
                input_dtype='float16',
                expected_output_shape=[None, 16, 3],
                expected_output_dtype='float16'
            )


@test_combinations.run_all_keras_modes
class TestDropBlock(test_combinations.TestCase):
    def setUp(self):
        super(TestDropBlock, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestDropBlock, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            DropBlock,
            kwargs={'rate': 0., 'size': 2},
            input_shape=[2, 16, 3, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 3, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            DropBlock,
            kwargs={'rate': 0.2, 'size': 1},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            DropBlock,
            kwargs={'rate': 0.1, 'size': 3},
            input_shape=[2, 16, 3, 1],
            input_dtype='float16',
            expected_output_shape=[None, 16, 3, 1],
            expected_output_dtype='float16'
        )

    def test_mean(self):
        inputs = np.random.uniform(size=[4, 32, 32, 3]).astype('float32')

        outputs = DropBlock(0.2, 3)(inputs, training=True)
        outputs = self.evaluate(outputs)

        self.assertNotAllClose(inputs, outputs)
        self.assertAllClose(inputs.mean(axis=(1, 2)), outputs.mean(axis=(1, 2)), atol=5e-2)


if __name__ == '__main__':
    tf.test.main()
