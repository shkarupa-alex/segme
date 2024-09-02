import numpy as np
from keras.src import layers
from keras.src import testing
from keras.src.applications.efficientnet_v2 import EfficientNetV2S

from segme.common.hmsattn import HierarchicalMultiScaleAttention


class TestHierarchicalMultiScaleAttention(testing.TestCase):
    def test_layer(self):
        model = EfficientNetV2S(
            input_tensor=layers.Input(
                name="image", shape=(None, None, 3), dtype="uint8"
            ),
            weights=None,
        )
        hmsa = HierarchicalMultiScaleAttention(
            model, "block3d_add", "block5i_add", (0.25, 0.5, 2.0)
        )

        inputs = np.zeros((2, 224, 224, 3), "uint8")
        result = hmsa(inputs)

        self.assertDType(result, "float32")
        self.assertListEqual(result.shape.as_list(), [2, 14, 14, 160])

        # TODO
        # self.run_layer_test(
        #     HierarchicalMultiScaleAttention,
        #     init_kwargs={
        #         "model": EfficientNetV2S(
        #             input_tensor=layers.Input(
        #                 name="image", shape=(None, None, 3), dtype="uint8"
        #             ),
        #             weights=None,
        #         ),
        #         "features": "block3d_add",
        #         "logits": "block5i_add",
        #         "scales": (0.25, 0.5, 2.0),
        #         "filters": 256,
        #         "dropout": 0.0,
        #     },
        #     input_shape=(2, 256, 256, 3),
        #     input_dtype="uint8",
        #     expected_output_shape=(2, 8, 8, 2),
        #     expected_output_dtype="float32",
        # )
