import numpy as np
import tensorflow as tf
from keras import keras_parameterized
from keras.mixed_precision import policy as mixed_precision
from keras.utils.losses_utils import ReductionV2 as Reduction
from ..loss import PointLoss
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestPointLoss(keras_parameterized.TestCase):
    def setUp(self):
        super(TestPointLoss, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestPointLoss, self).tearDown()
        mixed_precision.set_policy(self.default_policy)

    def test_layer(self):
        outputs = layer_multi_io_test(
            PointLoss,
            kwargs={'classes': 5, 'weighted': False, 'reduction': Reduction.NONE},
            input_datas=[
                np.random.rand(2, 16, 5),
                np.random.rand(2, 16, 2),
                (np.random.rand(2, 8, 8, 1) > 0.5).astype(np.int32)
            ],
            input_dtypes=['float32', 'float32', 'int32'],
            expected_output_shapes=[(None, 16)],
            expected_output_dtypes=['float32']
        )
        self.assertTrue(np.all(outputs >= 0.))

        outputs = layer_multi_io_test(
            PointLoss,
            kwargs={'classes': 5, 'weighted': True, 'reduction': Reduction.NONE},
            input_datas=[
                np.random.rand(2, 16, 5),
                np.random.rand(2, 16, 2),
                (np.random.rand(2, 8, 8, 1) > 0.5).astype(np.int32),
                np.random.rand(2, 8, 8, 1)
            ],
            input_dtypes=['float32', 'float32', 'int32', 'float32'],
            expected_output_shapes=[(None, 16)],
            expected_output_dtypes=['float32']
        )
        self.assertTrue(np.all(outputs >= 0.))

        mixed_precision.set_policy('mixed_float16')
        outputs = layer_multi_io_test(
            PointLoss,
            kwargs={'classes': 5, 'weighted': True, 'reduction': Reduction.NONE},
            input_datas=[
                np.random.rand(2, 16, 5),
                np.random.rand(2, 16, 2),
                (np.random.rand(2, 8, 8, 1) > 0.5).astype(np.int32),
                np.random.rand(2, 8, 8, 1)
            ],
            input_dtypes=['float32', 'float32', 'int32', 'float32'],
            expected_output_shapes=[(None, 16)],
            expected_output_dtypes=['float32']
        )
        self.assertTrue(np.all(outputs >= 0.))

        outputs = layer_multi_io_test(
            PointLoss,
            kwargs={'classes': 5, 'weighted': True, 'reduction': Reduction.NONE},
            input_datas=[
                np.random.rand(2, 16, 5).astype(np.float16),
                np.random.rand(2, 16, 2).astype(np.float16),
                (np.random.rand(2, 8, 8, 1) > 0.5).astype(np.int32),
                np.random.rand(2, 8, 8, 1).astype(np.float16)
            ],
            input_dtypes=['float16', 'float16', 'int32', 'float16'],
            expected_output_shapes=[(None, 16)],
            expected_output_dtypes=['float32']
        )
        self.assertTrue(np.all(outputs >= 0.))


if __name__ == '__main__':
    tf.test.main()
