from keras.src import testing

from segme.model.refinement.hqs_crm.encoder import Encoder


class TestEncoder(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            Encoder,
            init_kwargs={},
            input_shape=(2, 256, 256, 4),
            input_dtype="uint8",
            expected_output_shape=(
                (2, 128, 128, 64),
                (2, 64, 64, 256),
                (2, 32, 32, 2048),
            ),
            expected_output_dtype=("float32",) * 3,
        )
