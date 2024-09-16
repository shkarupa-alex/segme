import numpy as np
from keras.src import testing

from segme.model.matting.fba_matting.loss import fba_matting_losses
from segme.model.matting.fba_matting.model import FBAMatting


class TestFBAMatting(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            FBAMatting,
            init_kwargs={},
            input_shape=((2, 120, 120, 3), (2, 120, 120, 2), (2, 120, 120, 6)),
            input_dtype=("uint8",) * 3,
            expected_output_shape=(
                (2, 120, 120, 7),
                (2, 120, 120, 1),
                (2, 120, 120, 3),
                (2, 120, 120, 3),
            ),
            expected_output_dtype=("float32",) * 4,
        )

    def test_model(self):
        model = FBAMatting()
        model.compile(optimizer="sgd", loss=fba_matting_losses())
        model.fit(
            [
                np.random.random((2, 240, 240, 3)).astype(np.uint8),
                np.random.random((2, 240, 240, 2)).astype(np.uint8),
                np.random.random((2, 240, 240, 6)).astype(np.uint8),
            ],
            [
                np.random.random((2, 240, 240, 7)).astype(np.float32),
                np.random.random((2, 240, 240, 1)).astype(np.float32),
                np.random.random((2, 240, 240, 3)).astype(np.float32),
                np.random.random((2, 240, 240, 3)).astype(np.float32),
            ],
            epochs=1,
            batch_size=10,
        )

        # test config
        model.get_config()
