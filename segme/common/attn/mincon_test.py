import numpy as np
from keras.src import backend
from keras.src import testing

from segme.common.attn.mincon import MinConstraint


class TestDeformableConstraint(testing.TestCase):
    def test_value(self):
        kernel = np.random.uniform(size=(16,)).astype("float32")
        result = MinConstraint(0.2)(kernel)
        result = backend.convert_to_numpy(result)
        self.assertTrue(result.min() >= 0.0)
        self.assertTrue(result.max() <= 0.2 + 1e-6)
