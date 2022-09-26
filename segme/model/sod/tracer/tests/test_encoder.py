import numpy as np
import tensorflow as tf
from keras.testing_infra import test_combinations
from segme.model.sod.tracer.encoder import Encoder
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestEncoder(test_combinations.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            Encoder,
            kwargs={'radius': 5, 'confidence': 0.1},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['float32'],
            expected_output_shapes=[
                (None, 56, 56, 1),
                (None, 56, 56, 256),
                (None, 28, 28, 512),
                (None, 14, 14, 1024),
                (None, 7, 7, 2048)
            ],
            expected_output_dtypes=['float32'] * 5
        )


if __name__ == '__main__':
    tf.test.main()
