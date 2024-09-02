from keras.src import testing

from segme.model.refinement.cascade_psp.encoder import Encoder


class TestEncoder(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            Encoder,
            init_kwargs={},
            input_shape=(2, 512, 512, 6),
            input_dtype="float32",
            expected_output_shape=(
                (2, 256, 256, 64),
                (2, 128, 128, 256),
                (2, 64, 64, 2048),
            ),
            expected_output_dtype=("float32",) * 3,
            run_mixed_precision_check=False, # TODO
        )
