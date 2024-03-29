import numpy as np
import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from segme.model.matting.fba_matting.twomap import Twomap


@test_combinations.run_all_keras_modes
class TestTwomap(test_combinations.TestCase):
    def test_layer(self):
        result = test_utils.layer_test(
            Twomap,
            kwargs={},
            input_shape=[2, 64, 64, 1],
            input_dtype='uint8',
            expected_output_shape=[None, 64, 64, 2],
            expected_output_dtype='float32'
        )
        self.assertAllLessEqual(result, 255.)
        self.assertAllGreaterEqual(result, 0.)

    def test_value(self):
        trimap = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 0, 128, 0, 0],
            [0, 0, 0, 0, 0, 0, 128, 0, 128, 128, 128, 128, 128, 0, 128, 0],
            [0, 0, 0, 0, 0, 128, 0, 0, 0, 128, 128, 128, 128, 0, 0, 128],
            [0, 0, 0, 0, 0, 128, 128, 128, 0, 255, 128, 128, 128, 128, 0, 128],
            [0, 0, 0, 0, 0, 0, 128, 128, 128, 255, 128, 128, 128, 128, 0, 128],
            [0, 0, 0, 0, 0, 0, 128, 128, 255, 255, 128, 255, 255, 128, 128, 0],
            [0, 0, 0, 0, 0, 0, 0, 255, 128, 255, 255, 128, 255, 255, 128, 0],
            [0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 255, 255, 128, 128, 0],
            [0, 0, 0, 0, 0, 0, 128, 0, 128, 128, 128, 128, 128, 128, 128, 0],
            [0, 0, 0, 0, 128, 128, 0, 0, 255, 128, 128, 128, 255, 128, 255, 0],
            [0, 0, 0, 0, 0, 0, 128, 0, 255, 255, 128, 255, 128, 128, 128, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 128, 128, 128, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 128, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 255, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 128, 255, 255, 255, 128, 0, 0, 128, 0]
        ], 'uint8')
        expected = _twomap(trimap)
        expected = (expected * 255.).astype('uint8')

        result = Twomap()(trimap[None, ..., None])[0]
        result = self.evaluate(result)

        self.assertAllClose(expected, result)


def _twomap(trimap):
    twomap = np.zeros(trimap.shape[:2] + (2,), 'float32')
    twomap[trimap / 255.0 == 1, 1] = 1
    twomap[trimap / 255.0 == 0, 0] = 1

    return twomap


if __name__ == '__main__':
    tf.test.main()
