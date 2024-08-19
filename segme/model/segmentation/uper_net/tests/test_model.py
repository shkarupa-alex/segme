from keras.src import testing

from segme.model.segmentation.uper_net.model import UPerNet


class TestUPerNet(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            UPerNet,
            init_kwargs={
                "classes": 1,
                "decoder_filters": 8,
                "head_dropout": 0.0,
            },
            input_shape=(2, 240, 240, 3),
            input_dtype="uint8",
            expected_output_shape=(2, 240, 240, 1),
            expected_output_dtype="float32",
        )
