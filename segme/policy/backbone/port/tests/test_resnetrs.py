from absl.testing import parameterized
from keras.src import testing
from keras.src.applications import imagenet_utils
from keras.src.utils import file_utils
from keras.src.utils import image_utils

from segme.common.backbone import Backbone
from segme.policy import cnapol
from segme.policy.backbone.port.resnetrs import ResNetRS50


class TestResNetRS(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        super(TestResNetRS, self).setUp()
        self.default_cna = cnapol.global_policy()

    def tearDown(self):
        super(TestResNetRS, self).tearDown()
        cnapol.set_global_policy(self.default_cna)

    def test_port(self):
        test_image = file_utils.get_file(
            "elephant.jpg",
            "https://storage.googleapis.com/tensorflow/"
            "keras-applications/tests/elephant.jpg",
        )
        image = image_utils.load_img(
            test_image, target_size=(224, 224), interpolation="bicubic"
        )
        image = image_utils.img_to_array(image)[None, ...]

        ported = ResNetRS50()
        preds = ported.predict(image)

        names = [
            p[1] for p in imagenet_utils.decode_predictions(preds, top=3)[0]
        ]
        self.assertIn("African_elephant", names)

    # def test_deserialize_policy(self):
    # TODO

    def test_50(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "resnet_rs_50-imagenet"},
            input_shape=(2, 224, 224, 3),
            input_dtype="uint8",
            expected_output_shape=(
                (2, 112, 112, 64),
                (2, 56, 56, 256),
                (2, 28, 28, 512),
                (2, 14, 14, 1024),
                (2, 7, 7, 2048),
            ),
            expected_output_dtype=("float32",) * 5,
        )

    @parameterized.parameters([101, 152, 200, 270, 350, 420])
    def test_rest(self, size):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": f"resnet_rs_{size}-none"},
            input_shape=(2, 224, 224, 3),
            input_dtype="uint8",
            expected_output_shape=(
                (2, 112, 112, 64),
                (2, 56, 56, 256),
                (2, 28, 28, 512),
                (2, 14, 14, 1024),
                (2, 7, 7, 2048),
            ),
            expected_output_dtype=("float32",) * 5,
        )

    @parameterized.parameters(["imagenet", "none"])
    def test_50_cna_policy(self, init):
        cnapol.set_global_policy("stdconv-gn-leakyrelu")
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": f"resnet_rs_50-{init}"},
            input_shape=(2, 224, 224, 3),
            input_dtype="uint8",
            expected_output_shape=(
                (2, 112, 112, 64),
                (2, 56, 56, 256),
                (2, 28, 28, 512),
                (2, 14, 14, 1024),
                (2, 7, 7, 2048),
            ),
            expected_output_dtype=("float32",) * 5,
        )

    def test_50_s8(self):
        self.run_layer_test(
            Backbone,
            init_kwargs={"scales": None, "policy": "resnet_rs_50_s8-imagenet"},
            input_shape=(2, 224, 224, 3),
            input_dtype="uint8",
            expected_output_shape=(
                (2, 112, 112, 64),
                (2, 56, 56, 256),
                (2, 28, 28, 512),
                (2, 28, 28, 1024),
                (2, 28, 28, 2048),
                # (2, 112, 112, 64),
                # (2, 56, 56, 256),
                # (2, 28, 28, 512),
                # (2, 14, 14, 1024),
                # (2, 7, 7, 2048),
            ),
            expected_output_dtype=("float32",) * 5,
        )
