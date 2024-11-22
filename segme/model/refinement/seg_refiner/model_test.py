import numpy as np
from keras.src import testing

from segme.model.refinement.seg_refiner.loss import seg_refiner_loss
from segme.model.refinement.seg_refiner.model import SegRefiner


class TestSegRefiner(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            SegRefiner,
            init_kwargs={
                "filters": 128,
                "depth": 2,
                "atstrides": (16, 32),
                "dropout": 0,
                "mults": (1, 1, 2, 2, 4, 4),
                "heads": 4,
            },
            input_shape=((2, 256, 256, 3), (2, 256, 256, 1), (2,)),
            input_dtype=("uint8", "uint8", "int32"),
            expected_output_shape=(2, 256, 256, 1),
            expected_output_dtype="float32",
        )

    def test_model(self):
        model = SegRefiner()
        model.compile(optimizer="sgd", loss=seg_refiner_loss())
        model.fit(
            [
                np.random.random((2, 256, 256, 3)).astype(np.uint8),
                np.random.random((2, 256, 256, 1)).astype(np.uint8),
                np.random.random((2,)).astype(np.uint8),
            ],
            np.random.random((2, 256, 256, 1)).astype(np.float32),
            epochs=1,
            batch_size=10,
        )

        # test config
        model.get_config()
