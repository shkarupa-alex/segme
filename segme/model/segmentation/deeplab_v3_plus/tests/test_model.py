from keras.src import testing

from segme.model.segmentation.deeplab_v3_plus.model import DeepLabV3Plus
from segme.model.segmentation.deeplab_v3_plus.model import DeepLabV3PlusHMS


class TestDeepLabV3Plus(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            DeepLabV3Plus,
            init_kwargs={
                "classes": 4,
                "aspp_filters": 8,
                "aspp_stride": 32,
                "low_filters": 16,
                "decoder_filters": 4,
            },
            input_shape=(2, 224, 224, 3),
            input_dtype="uint8",
            expected_output_shape=(2, 224, 224, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            DeepLabV3PlusHMS,
            init_kwargs={
                "classes": 4,
                "aspp_filters": 8,
                "aspp_stride": 32,
                "low_filters": 16,
                "decoder_filters": 4,
            },
            input_shape=(2, 224, 224, 3),
            input_dtype="uint8",
            expected_output_shape=(2, 224, 224, 4),
            expected_output_dtype="float32",
        )
