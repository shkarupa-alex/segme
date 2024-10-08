import numpy as np
from keras.src import testing

from segme.utils.matting.compose import compose_batch
from segme.utils.matting.compose import compose_two
from segme.utils.matting.compose import random_compose
from segme.utils.matting_np.compose import compose_two as compose_two_np


class TestComposeTwo(testing.TestCase):
    def test_compose(self):
        fg = np.random.uniform(0.0, 255.0, (2, 16, 16, 3)).astype("uint8")
        alpha = np.random.uniform(0.0, 255.0, (2, 16, 16, 1)).astype("uint8")

        fg_, alpha_ = compose_two(fg, alpha)

        self.assertEqual(fg_.dtype, "uint8")
        self.assertEqual(alpha_.dtype, "uint8")
        self.assertListEqual(fg_.shape.as_list(), [1, 16, 16, 3])
        self.assertListEqual(alpha_.shape.as_list(), [1, 16, 16, 1])

        expected_fg, expected_alpha = compose_two_np(
            fg[0], alpha[0], fg[1], alpha[1]
        )

        error_fg = np.abs(expected_fg - fg_[0]).mean() / 255.0
        self.assertLess(error_fg, 0.6)
        self.assertAlmostEqual(alpha_[0], expected_alpha)

    def test_drop(self):
        fg = np.random.uniform(0.0, 255.0, (4, 16, 16, 3)).astype("uint8")
        alpha = np.random.uniform(0.0, 255.0, (4, 16, 16, 1)).astype("uint8")
        alpha[0] = 255
        alpha[2] = 255

        fg_, alpha_ = compose_two(fg, alpha, solve=False)

        self.assertEqual(fg_.dtype, "uint8")
        self.assertListEqual(fg_.shape.as_list(), [1, 16, 16, 3])

        self.assertEqual(alpha_.dtype, "uint8")
        self.assertListEqual(alpha_.shape.as_list(), [1, 16, 16, 1])


class TestComposeBatch(testing.TestCase):
    def test_compose(self):
        fg = np.random.uniform(0.0, 255.0, (5, 16, 16, 3)).astype("uint8")
        alpha = np.random.uniform(0.0, 255.0, (5, 16, 16, 1)).astype("uint8")

        fg_, alpha_ = compose_batch(fg, alpha, solve=False)

        self.assertEqual(fg_.dtype, "uint8")
        self.assertNotAllClose(fg_, fg)

        self.assertEqual(alpha_.dtype, "uint8")
        self.assertNotAllClose(alpha_, alpha)

    def test_solve(self):
        fg = np.random.uniform(0.0, 255.0, (4, 16, 16, 3)).astype("uint8")
        alpha = np.random.uniform(0.0, 255.0, (4, 16, 16, 1)).astype("uint8")

        fg_, alpha_ = compose_batch(fg, alpha, solve=True)

        self.assertEqual(fg_.dtype, "uint8")
        self.assertEqual(alpha_.dtype, "uint8")
        self.assertListEqual(fg_.shape.as_list(), [4, 16, 16, 3])
        self.assertListEqual(alpha_.shape.as_list(), [4, 16, 16, 1])


class TestRandomCompose(testing.TestCase):
    def test_no_compose(self):
        fg = np.random.uniform(0.0, 255.0, (4, 16, 16, 3)).astype("uint8")
        alpha = np.random.uniform(0.0, 255.0, (4, 16, 16, 1)).astype("uint8")

        fg_, alpha_ = random_compose(fg, alpha, prob=0.0, solve=False)

        self.assertEqual(fg_.dtype, "uint8")
        self.assertAlmostEqual(fg_, fg)

        self.assertEqual(alpha_.dtype, "uint8")
        self.assertAlmostEqual(alpha_, alpha)

    def test_compose(self):
        fg = np.random.uniform(0.0, 255.0, (5, 16, 16, 3)).astype("uint8")
        alpha = np.random.uniform(0.0, 255.0, (5, 16, 16, 1)).astype("uint8")

        fg_, alpha_ = random_compose(fg, alpha, prob=1.0, solve=False)

        self.assertEqual(fg_.dtype, "uint8")
        self.assertNotAllClose(fg_, fg)

        self.assertEqual(alpha_.dtype, "uint8")
        self.assertNotAllClose(alpha_, alpha)

    def test_solve(self):
        fg = np.random.uniform(0.0, 255.0, (4, 16, 16, 3)).astype("uint8")
        alpha = np.random.uniform(0.0, 255.0, (4, 16, 16, 1)).astype("uint8")

        fg_, alpha_ = random_compose(fg, alpha, solve=True)

        self.assertEqual(fg_.dtype, "uint8")
        self.assertEqual(alpha_.dtype, "uint8")
        self.assertListEqual(fg_.shape.as_list(), [4, 16, 16, 3])
        self.assertListEqual(alpha_.shape.as_list(), [4, 16, 16, 1])
