import numpy as np
from keras.src import testing

from segme.model.refinement.hqs_crm.loss import hqs_crm_loss
from segme.model.refinement.hqs_crm.model import HqsCrm
from segme.model.refinement.hqs_crm.model import build_hqs_crm


class TestHqsCrm(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            HqsCrm,
            init_kwargs={
                "aspp_filters": (64, 64, 128),
                "aspp_drop": 0.5,
                "mlp_units": (32, 32, 32, 32),
            },
            input_shape=((2, 224, 224, 3), (2, 224, 224, 1), (2, 224, 224, 2)),
            input_dtype=("uint8", "uint8", "float32"),
            expected_output_shape=(2, 224, 224, 1),
            expected_output_dtype="float32",
        )

    def test_model(self):
        model = build_hqs_crm(
            aspp_filters=(64, 64, 128),
            aspp_drop=0.5,
            mlp_units=(32, 32, 32, 32),
        )
        model.compile(
            optimizer="sgd",
            loss=hqs_crm_loss(),
        )
        model.fit(
            [
                np.random.random((2, 224, 224, 3)).astype(np.uint8),
                np.random.random((2, 224, 224, 1)).astype(np.uint8),
                np.random.random((2, 224, 224, 2)).astype(np.float32),
            ],
            np.random.randint(0, 1, (2, 224, 224, 1)),
            epochs=1,
            batch_size=10,
        )

        # test config
        model.get_config()
