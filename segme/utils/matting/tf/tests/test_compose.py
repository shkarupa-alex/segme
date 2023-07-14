import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from segme.utils.matting.tf.compose import compose_two, compose_batch, random_compose
from segme.utils.matting.np.compose import compose_two as compose_two_np


@test_util.run_all_in_graph_and_eager_modes
class TestComposeTwo(tf.test.TestCase):
    def test_compose(self):
        fg = np.random.uniform(0., 255., (2, 16, 16, 3)).astype('uint8')
        alpha = np.random.uniform(0., 255., (2, 16, 16, 1)).astype('uint8')

        result = compose_two(fg, alpha, solve=False)
        fg_, alpha_ = self.evaluate(result)

        expected_fg, expected_alpha = compose_two_np(fg[0], alpha[0], fg[1], alpha[1], solve=False)

        self.assertEqual(fg_.dtype, 'uint8')
        self.assertAllEqual(fg_[0], expected_fg)

        self.assertEqual(alpha_.dtype, 'uint8')
        self.assertAllEqual(alpha_[0], expected_alpha)

    def test_drop(self):
        fg = np.random.uniform(0., 255., (4, 16, 16, 3)).astype('uint8')
        alpha = np.random.uniform(0., 255., (4, 16, 16, 1)).astype('uint8')
        alpha[0] = 255
        alpha[2] = 255

        result = compose_two(fg, alpha, solve=False)
        fg_, alpha_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, 'uint8')
        self.assertTupleEqual(fg_.shape, (1, 16, 16, 3))

        self.assertEqual(alpha_.dtype, 'uint8')
        self.assertTupleEqual(alpha_.shape, (1, 16, 16, 1))

    def test_solve(self):
        fg = np.random.uniform(0., 255., (2, 16, 16, 3)).astype('uint8')
        alpha = np.random.uniform(0., 255., (2, 16, 16, 1)).astype('uint8')

        result = compose_two(fg, alpha)
        fg_, alpha_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, 'uint8')
        self.assertEqual(alpha_.dtype, 'uint8')
        self.assertTupleEqual(fg_.shape, (1, 16, 16, 3))
        self.assertTupleEqual(alpha_.shape, (1, 16, 16, 1))

        expected_fg, expected_alpha = compose_two_np(fg[0], alpha[0], fg[1], alpha[1])

        error_fg = np.abs(expected_fg - fg_[0]).mean() / 255.
        self.assertLess(error_fg, 0.6)
        self.assertAllEqual(alpha_[0], expected_alpha)


@test_util.run_all_in_graph_and_eager_modes
class TestComposeBatch(tf.test.TestCase):
    def test_compose(self):
        fg = np.random.uniform(0., 255., (5, 16, 16, 3)).astype('uint8')
        alpha = np.random.uniform(0., 255., (5, 16, 16, 1)).astype('uint8')

        result = compose_batch(fg, alpha, solve=False)
        fg_, alpha_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, 'uint8')
        self.assertNotAllEqual(fg_, fg)

        self.assertEqual(alpha_.dtype, 'uint8')
        self.assertNotAllEqual(alpha_, alpha)

    def test_solve(self):
        fg = np.random.uniform(0., 255., (4, 16, 16, 3)).astype('uint8')
        alpha = np.random.uniform(0., 255., (4, 16, 16, 1)).astype('uint8')

        result = compose_batch(fg, alpha, solve=True)
        fg_, alpha_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, 'uint8')
        self.assertEqual(alpha_.dtype, 'uint8')
        self.assertTupleEqual(fg_.shape, (4, 16, 16, 3))
        self.assertTupleEqual(alpha_.shape, (4, 16, 16, 1))


@test_util.run_all_in_graph_and_eager_modes
class TestRandomCompose(tf.test.TestCase):
    def test_no_compose(self):
        fg = np.random.uniform(0., 255., (4, 16, 16, 3)).astype('uint8')
        alpha = np.random.uniform(0., 255., (4, 16, 16, 1)).astype('uint8')

        result = random_compose(fg, alpha, prob=0., solve=False)
        fg_, alpha_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, 'uint8')
        self.assertAllEqual(fg_, fg)

        self.assertEqual(alpha_.dtype, 'uint8')
        self.assertAllEqual(alpha_, alpha)

    def test_compose(self):
        fg = np.random.uniform(0., 255., (5, 16, 16, 3)).astype('uint8')
        alpha = np.random.uniform(0., 255., (5, 16, 16, 1)).astype('uint8')

        result = random_compose(fg, alpha, prob=1., solve=False)
        fg_, alpha_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, 'uint8')
        self.assertNotAllEqual(fg_, fg)

        self.assertEqual(alpha_.dtype, 'uint8')
        self.assertNotAllEqual(alpha_, alpha)

    def test_solve(self):
        fg = np.random.uniform(0., 255., (4, 16, 16, 3)).astype('uint8')
        alpha = np.random.uniform(0., 255., (4, 16, 16, 1)).astype('uint8')

        result = random_compose(fg, alpha, solve=True)
        fg_, alpha_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, 'uint8')
        self.assertEqual(alpha_.dtype, 'uint8')
        self.assertTupleEqual(fg_.shape, (4, 16, 16, 3))
        self.assertTupleEqual(alpha_.shape, (4, 16, 16, 1))


if __name__ == '__main__':
    tf.test.main()
