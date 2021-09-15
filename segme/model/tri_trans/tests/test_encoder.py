import numpy as np
import tensorflow as tf
from keras import keras_parameterized
from keras.applications import resnet
from ..encoder import MMFusionEncoder
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestMMFusionEncoder(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            MMFusionEncoder,
            kwargs={},
            input_shapes=[(2, 256, 256, 3), (2, 256, 256, 1)],
            input_dtypes=['uint8', 'uint16'],
            expected_output_shapes=[
                (None, 128, 128, 64),
                (None, 64, 64, 256),
                (None, 32, 32, 512),
                (None, 16, 16, 1024),
                (None, 8, 8, 2048)],
            expected_output_dtypes=['float32'] * 5
        )

    def test_rgb_slice(self):
        source = np.random.randint(0, 64, (2 * 224, 224 * 3), 'uint8')
        np.fill_diagonal(source, 128)
        source = np.rot90(source)
        np.fill_diagonal(source, 256)
        source = np.reshape(source, (2, 224, 224, 3))

        expected = resnet.preprocess_input(source)
        expected = resnet.ResNet50(input_shape=[224, 224, 3], include_top=False, weights='imagenet')(expected)
        expected = self.evaluate(expected)

        layer = MMFusionEncoder()
        layer.build([(None, None, None, 3), (None, None, None, 1)])

        result = tf.cast(source, 'float32')
        result = result[..., ::-1]  # 'RGB'->'BGR'
        result = tf.nn.bias_add(result, [-103.939, -116.779, -123.680])
        result = layer.rgb_bone2(result)
        result = layer.rgb_bone4(result)
        result = layer.rgb_bone8(result)
        result = layer.rgb_bone16(result)
        result = layer.rgb_bone32(result)
        result = self.evaluate(result)

        self.assertTrue(np.all(expected == result))


if __name__ == '__main__':
    tf.test.main()
