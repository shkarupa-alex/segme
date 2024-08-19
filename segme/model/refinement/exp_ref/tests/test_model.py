import numpy as np
from keras.src import testing

from segme.model.refinement.exp_ref.model import ExpRef


class TestExpRef(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            ExpRef,
            init_kwargs={"sup_unfold": False},
            input_shape=((2, 240, 240, 3), (2, 240, 240, 1)),
            input_dtype=("uint8",) * 2,
            expected_output_shape=((2, 240, 240, 1),) * 3,
            expected_output_dtype=("float32",) * 3,
        )
        self.run_layer_test(
            ExpRef,
            init_kwargs={"sup_unfold": True},
            input_shape=((2, 240, 240, 3), (2, 240, 240, 1)),
            input_dtype=("uint8",) * 2,
            expected_output_shape=((2, 240, 240, 1),) * 3,
            expected_output_dtype=("float32",) * 3,
        )

    def test_model(self):
        model = ExpRef()
        model.compile(
            optimizer="sgd",
            loss="mse",
            # jit_compile=False, # TODO
        )
        model.fit(
            [
                np.random.random((2, 240, 240, 3)).astype(np.uint8),
                np.random.random((2, 240, 240, 1)).astype(np.uint8),
            ],
            np.random.random((2, 240, 240, 1)).astype(np.float32),
            epochs=1,
            batch_size=10,
        )

        # test config
        model.get_config()
