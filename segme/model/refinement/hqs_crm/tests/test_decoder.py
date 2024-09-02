from keras.src import testing

from segme.model.refinement.hqs_crm.decoder import Decoder


class TestDecoder(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            Decoder,
            init_kwargs={
                "mlp_units": (32, 32, 32, 32),
            },
            input_shape=(
                (3, 128, 128, 64),
                (3, 96, 96, 2),
            ),
            input_dtype=("float32",) * 2,
            expected_output_shape=(3, 96, 96, 1),
            expected_output_dtype="float32",
        )
