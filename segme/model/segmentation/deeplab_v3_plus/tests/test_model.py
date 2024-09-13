import numpy as np
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

        model = DeepLabV3PlusHMS(
            classes=4,
            aspp_filters=8,
            aspp_stride=32,
            low_filters=16,
            decoder_filters=4,
            scales=(0.25, 0.5, 2.0),
        )

        inputs = np.zeros((2, 224, 224, 3), "uint8")
        result = model(inputs)

        self.assertDType(result, "float32")
        self.assertListEqual(result.shape.as_list(), [2, 224, 224, 4])
