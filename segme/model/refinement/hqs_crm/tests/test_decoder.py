from keras.src import testing

from segme.model.refinement.hqs_crm.decoder import Decoder


class TestDecoder(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            Decoder,
            init_kwargs={
                "aspp_filters": (64, 64, 128),
                "aspp_drop": 0.5,
                "mlp_units": (32, 32, 32, 32),
            },
            input_shape=(
                (3, 128, 128, 64),
                (3, 64, 64, 256),
                (3, 32, 32, 2048),
                (3, 96, 96, 2),
            ),
            input_dtype=("float32",) * 4,
            expected_output_shape=(2, 96, 96, 1),
            expected_output_dtype="float32",
        )
