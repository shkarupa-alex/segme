from keras.src import testing

from segme.model.sod.u2_net.model import U2Net
from segme.model.sod.u2_net.model import U2NetP


class TestU2Net(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            U2Net,
            init_kwargs={"classes": 1},
            input_shape=(2, 64, 64, 3),
            input_dtype="uint8",
            expected_output_shape=((2, 64, 64, 1),) * 7,
            expected_output_dtype=("float32",) * 7,
        )
        self.run_layer_test(
            U2NetP,
            init_kwargs={"classes": 3},
            input_shape=(2, 64, 64, 3),
            input_dtype="uint8",
            expected_output_shape=((2, 64, 64, 3),) * 7,
            expected_output_dtype=("float32",) * 7,
        )
