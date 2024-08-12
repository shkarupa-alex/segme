import numpy as np
import tensorflow as tf

from segme.utils.matting.np.compose import compose_two as compose_two_np
from segme.utils.matting.tf.compose import compose_batch
from segme.utils.matting.tf.compose import compose_two
from segme.utils.matting.tf.compose import random_compose


class TestComposeTwo(tf.test.TestCase):
    def test_compose(self):
        fg = np.random.uniform(0.0, 255.0, (2, 16, 16, 3)).astype("uint8")
        alpha = np.random.uniform(0.0, 255.0, (2, 16, 16, 1)).astype("uint8")

        result = compose_two(fg, alpha, solve=False)
        fg_, alpha_ = self.evaluate(result)

        expected_fg, expected_alpha = compose_two_np(
            fg[0], alpha[0], fg[1], alpha[1], solve=False
        )

        self.assertEqual(fg_.dtype, "uint8")
        self.assertAllEqual(fg_[0], expected_fg)

        self.assertEqual(alpha_.dtype, "uint8")
        self.assertAllEqual(alpha_[0], expected_alpha)

    def test_drop(self):
        fg = np.random.uniform(0.0, 255.0, (4, 16, 16, 3)).astype("uint8")
        alpha = np.random.uniform(0.0, 255.0, (4, 16, 16, 1)).astype("uint8")
        alpha[0] = 255
        alpha[2] = 255

        result = compose_two(fg, alpha, solve=False)
        fg_, alpha_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, "uint8")
        self.assertTupleEqual(fg_.shape, (1, 16, 16, 3))

        self.assertEqual(alpha_.dtype, "uint8")
        self.assertTupleEqual(alpha_.shape, (1, 16, 16, 1))

    def test_solve(self):
        fg = np.random.uniform(0.0, 255.0, (2, 16, 16, 3)).astype("uint8")
        alpha = np.random.uniform(0.0, 255.0, (2, 16, 16, 1)).astype("uint8")

        result = compose_two(fg, alpha)
        fg_, alpha_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, "uint8")
        self.assertEqual(alpha_.dtype, "uint8")
        self.assertTupleEqual(fg_.shape, (1, 16, 16, 3))
        self.assertTupleEqual(alpha_.shape, (1, 16, 16, 1))

        expected_fg, expected_alpha = compose_two_np(
            fg[0], alpha[0], fg[1], alpha[1]
        )

        error_fg = np.abs(expected_fg - fg_[0]).mean() / 255.0
        self.assertLess(error_fg, 0.6)
        self.assertAllEqual(alpha_[0], expected_alpha)


class TestComposeBatch(tf.test.TestCase):
    def test_compose(self):
        fg = np.random.uniform(0.0, 255.0, (5, 16, 16, 3)).astype("uint8")
        alpha = np.random.uniform(0.0, 255.0, (5, 16, 16, 1)).astype("uint8")

        result = compose_batch(fg, alpha, solve=False)
        fg_, alpha_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, "uint8")
        self.assertNotAllEqual(fg_, fg)

        self.assertEqual(alpha_.dtype, "uint8")
        self.assertNotAllEqual(alpha_, alpha)

    def test_solve(self):
        fg = np.random.uniform(0.0, 255.0, (4, 16, 16, 3)).astype("uint8")
        alpha = np.random.uniform(0.0, 255.0, (4, 16, 16, 1)).astype("uint8")

        result = compose_batch(fg, alpha, solve=True)
        fg_, alpha_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, "uint8")
        self.assertEqual(alpha_.dtype, "uint8")
        self.assertTupleEqual(fg_.shape, (4, 16, 16, 3))
        self.assertTupleEqual(alpha_.shape, (4, 16, 16, 1))


class TestRandomCompose(tf.test.TestCase):
    def test_no_compose(self):
        fg = np.random.uniform(0.0, 255.0, (4, 16, 16, 3)).astype("uint8")
        alpha = np.random.uniform(0.0, 255.0, (4, 16, 16, 1)).astype("uint8")

        result = random_compose(fg, alpha, prob=0.0, solve=False)
        fg_, alpha_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, "uint8")
        self.assertAllEqual(fg_, fg)

        self.assertEqual(alpha_.dtype, "uint8")
        self.assertAllEqual(alpha_, alpha)

    def test_compose(self):
        fg = np.random.uniform(0.0, 255.0, (5, 16, 16, 3)).astype("uint8")
        alpha = np.random.uniform(0.0, 255.0, (5, 16, 16, 1)).astype("uint8")

        result = random_compose(fg, alpha, prob=1.0, solve=False)
        fg_, alpha_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, "uint8")
        self.assertNotAllEqual(fg_, fg)

        self.assertEqual(alpha_.dtype, "uint8")
        self.assertNotAllEqual(alpha_, alpha)

    def test_solve(self):
        fg = np.random.uniform(0.0, 255.0, (4, 16, 16, 3)).astype("uint8")
        alpha = np.random.uniform(0.0, 255.0, (4, 16, 16, 1)).astype("uint8")

        result = random_compose(fg, alpha, solve=True)
        fg_, alpha_ = self.evaluate(result)

        self.assertEqual(fg_.dtype, "uint8")
        self.assertEqual(alpha_.dtype, "uint8")
        self.assertTupleEqual(fg_.shape, (4, 16, 16, 3))
        self.assertTupleEqual(alpha_.shape, (4, 16, 16, 1))
