from keras.src import testing

from segme.model.refinement.cascade_psp.upsample import Upsample


class TestUpsample(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            Upsample,
            init_kwargs={"filters": 5},
            input_shape=((2, 4, 4, 3), (2, 16, 16, 3)),
            input_dtype=["float32", "float32"],
            expected_output_shape=((None, 16, 16, 5)),
            expected_output_dtype=["float32"],
        )
