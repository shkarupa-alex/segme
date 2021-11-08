import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from ..compose import compose_two
from ...np.compose import compose_two as compose_two_np


@test_util.run_all_in_graph_and_eager_modes
class TestComposeTwo(tf.test.TestCase):
    def test_no_compose(self):
        fg = np.random.uniform(0., 255., (2, 16, 16, 3)).astype('uint8')
        alpha = np.random.uniform(0., 255., (2, 16, 16, 1)).astype('uint8')

        result = compose_two(fg, alpha, prob=0.)
        fg__, alpha__ = self.evaluate(result)

        self.assertEqual(fg__.dtype, 'uint8')
        self.assertEqual(alpha__.dtype, 'uint8')
        self.assertAllEqual(fg__, fg)
        self.assertAllEqual(alpha__, alpha)

    def test_compose(self):
        fg = np.random.uniform(0., 255., (2, 16, 16, 3)).astype('uint8')
        alpha = np.random.uniform(0., 255., (2, 16, 16, 1)).astype('uint8')

        result = compose_two(fg, alpha, prob=1.)
        fg__, alpha__ = self.evaluate(result)

        self.assertEqual(fg__.dtype, 'uint8')
        self.assertEqual(alpha__.dtype, 'uint8')
        self.assertTupleEqual(fg__.shape, (1, 16, 16, 3))
        self.assertTupleEqual(alpha__.shape, (1, 16, 16, 1))

        expected_fg, expected_alpha = compose_two_np(fg[0], alpha[0], fg[1], alpha[1])
        self.assertAllEqual(fg__[0], expected_fg)
        self.assertAllEqual(alpha__[0], expected_alpha)

    def test_compose_solve(self):
        fg = np.random.uniform(0., 255., (2, 16, 16, 3)).astype('uint8')
        alpha = np.random.uniform(0., 255., (2, 16, 16, 1)).astype('uint8')

        result = compose_two(fg, alpha, prob=1., solve=True)
        fg__, alpha__ = self.evaluate(result)

        self.assertEqual(fg__.dtype, 'uint8')
        self.assertEqual(alpha__.dtype, 'uint8')
        self.assertTupleEqual(fg__.shape, (1, 16, 16, 3))
        self.assertTupleEqual(alpha__.shape, (1, 16, 16, 1))

        expected_fg, expected_alpha = compose_two_np(fg[0], alpha[0], fg[1], alpha[1])

        error_fg = np.abs(expected_fg - fg__[0]).mean() / 255.
        self.assertLess(error_fg, 0.6)
        self.assertAllEqual(alpha__[0], expected_alpha)


if __name__ == '__main__':
    tf.test.main()
