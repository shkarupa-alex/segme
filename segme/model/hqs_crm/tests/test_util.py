import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from ..util import make_coord


@test_util.run_all_in_graph_and_eager_modes
class TestMakeCoord(tf.test.TestCase):
    def test_value(self):
        expected = np.array([
            [[-0.8, -0.833333], [-0.8, -0.5], [-0.8, -0.166667], [-0.8, 0.166667], [-0.8, 0.5], [-0.8, 0.833333]],
            [[-0.4, -0.833333], [-0.4, -0.5], [-0.4, -0.166667], [-0.4, 0.166667], [-0.4, 0.5], [-0.4, 0.833333]],
            [[0., -0.833333], [0., -0.5], [0., -0.166667], [0., 0.166667], [0., 0.5], [0., 0.833333]],
            [[0.4, -0.833333], [0.4, -0.5], [0.4, -0.166667], [0.4, 0.166667], [0.4, 0.5], [0.4, 0.833333]],
            [[0.8, -0.833333], [0.8, -0.5], [0.8, -0.166667], [0.8, 0.166667], [0.8, 0.5], [0.8, 0.833333]]
        ])[None].repeat(3, axis=0)

        result = make_coord(3, 5, 6)
        result = self.evaluate(result)

        self.assertAllClose(expected, result)


if __name__ == '__main__':
    tf.test.main()
