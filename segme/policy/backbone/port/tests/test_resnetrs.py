import tensorflow as tf
from absl.testing import parameterized
from keras.applications import resnet_rs
from keras import mixed_precision
from keras.src.utils import data_utils, image_utils
from keras.src.testing_infra import test_combinations
from segme.common.backbone import Backbone
from segme.policy import cnapol
from segme.policy.backbone.port.resnetrs import ResNetRS50
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestResNetRS(test_combinations.TestCase):
    def setUp(self):
        super(TestResNetRS, self).setUp()
        self.default_cna = cnapol.global_policy()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestResNetRS, self).tearDown()
        cnapol.set_global_policy(self.default_cna)
        mixed_precision.set_global_policy(self.default_policy)

    def test_port(self):
        test_image = data_utils.get_file(
            'elephant.jpg', 'https://storage.googleapis.com/tensorflow/keras-applications/tests/elephant.jpg')
        image = image_utils.load_img(test_image, target_size=(224, 224), interpolation='bicubic')
        image = image_utils.img_to_array(image)[None, ...]

        original, ported = resnet_rs.ResNetRS50(include_top=False), ResNetRS50(include_top=False)
        self.assertEqual(len(original.weights), len(ported.weights))

        original.trainable = False
        ported.trainable = False
        original_preds = original.predict(image)
        ported_preds = ported.predict(image)

        self.assertAllClose(original_preds, ported_preds, atol=8e-6)

    def test_50(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'resnet_rs_50-imagenet'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 64),
                (None, 56, 56, 256),
                (None, 28, 28, 512),
                (None, 14, 14, 1024),
                (None, 7, 7, 2048)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    def test_50_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'resnet_rs_50-imagenet'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 64),
                (None, 56, 56, 256),
                (None, 28, 28, 512),
                (None, 14, 14, 1024),
                (None, 7, 7, 2048)
            ],
            expected_output_dtypes=['float16'] * 5
        )

    @parameterized.parameters([101, 152, 200, 270, 350, 420])
    def test_rest(self, size):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': f'resnet_rs_{size}-none'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 64),
                (None, 56, 56, 256),
                (None, 28, 28, 512),
                (None, 14, 14, 1024),
                (None, 7, 7, 2048)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    @parameterized.parameters(['imagenet', 'none'])
    def test_50_cna_policy(self, init):
        cnapol.set_global_policy('stdconv-gn-leakyrelu')
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': f'resnet_rs_50-{init}'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 64),
                (None, 56, 56, 256),
                (None, 28, 28, 512),
                (None, 14, 14, 1024),
                (None, 7, 7, 2048)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    def test_50_s8(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'resnet_rs_50_s8-imagenet'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 64),
                (None, 56, 56, 256),
                (None, 28, 28, 512),
                (None, 28, 28, 1024),
                (None, 28, 28, 2048)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    def test_50_s8_uniq_weights(self):
        # https://github.com/keras-team/keras/issues/18356
        cnapol.set_global_policy('stdconv-gn-leakyrelu')
        bb = Backbone(scales=[2, 4, 32], policy='resnet_rs_50_s8-imagenet')

        weights = [w.name for w in bb.weights]
        self.assertLen(set(weights), len(weights))

if __name__ == '__main__':
    tf.test.main()
