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
        fg_, alpha_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, 'uint8')
        self.assertAllEqual(fg_, fg)

        self.assertEqual(alpha_.dtype, 'uint8')
        self.assertAllEqual(alpha_, alpha)

    def test_compose(self):
        fg = np.random.uniform(0., 255., (2, 16, 16, 3)).astype('uint8')
        alpha = np.random.uniform(0., 255., (2, 16, 16, 1)).astype('uint8')

        result = compose_two(fg, alpha, prob=1.)
        fg_, alpha_ = self.evaluate(result)

        expected_fg, expected_alpha = compose_two_np(fg[0], alpha[0], fg[1], alpha[1])

        self.assertEqual(fg_.dtype, 'uint8')
        self.assertAllEqual(fg_[0], expected_fg)

        self.assertEqual(alpha_.dtype, 'uint8')
        self.assertAllEqual(alpha_[0], expected_alpha)

    def test_drop(self):
        fg = np.random.uniform(0., 255., (4, 16, 16, 3)).astype('uint8')
        alpha = np.random.uniform(0., 255., (4, 16, 16, 1)).astype('uint8')
        alpha[0] = 255
        alpha[2] = 255

        result = compose_two(fg, alpha, prob=1.)
        fg_, alpha_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, 'uint8')
        self.assertTupleEqual(fg_.shape, (1, 16, 16, 3))

        self.assertEqual(alpha_.dtype, 'uint8')
        self.assertTupleEqual(alpha_.shape, (1, 16, 16, 1))

    def test_rest(self):
        fg = np.random.uniform(0., 255., (8, 16, 16, 3)).astype('uint8')
        alpha = np.random.uniform(0., 255., (8, 16, 16, 1)).astype('uint8')
        rest = [np.zeros((8,)), np.ones((8, 3))]

        result = compose_two(fg, alpha, rest, prob=1.)
        fg_, alpha_, rest_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, 'uint8')
        self.assertTupleEqual(fg_.shape, (4, 16, 16, 3))

        self.assertEqual(alpha_.dtype, 'uint8')
        self.assertTupleEqual(alpha_.shape, (4, 16, 16, 1))

        for i in range(len(rest)):
            self.assertEqual(rest_[i].shape[0] * 2, rest[i].shape[0])
            self.assertEqual(rest_[i].dtype, rest[i].dtype)
            self.assertAllEqual(rest_[i], rest[i][:rest_[i].shape[0]])

    def test_solve(self):
        fg = np.random.uniform(0., 255., (2, 16, 16, 3)).astype('uint8')
        alpha = np.random.uniform(0., 255., (2, 16, 16, 1)).astype('uint8')

        result = compose_two(fg, alpha, prob=1., solve=True)
        fg_, alpha_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, 'uint8')
        self.assertEqual(alpha_.dtype, 'uint8')
        self.assertTupleEqual(fg_.shape, (1, 16, 16, 3))
        self.assertTupleEqual(alpha_.shape, (1, 16, 16, 1))

        expected_fg, expected_alpha = compose_two_np(fg[0], alpha[0], fg[1], alpha[1])

        error_fg = np.abs(expected_fg - fg_[0]).mean() / 255.
        self.assertLess(error_fg, 0.6)
        self.assertAllEqual(alpha_[0], expected_alpha)


if __name__ == '__main__':
    tf.test.main()
