import numpy as np
from keras.src import backend
from keras.src import testing

from segme.utils.matting.tf.aug import augment_alpha
from segme.utils.matting.tf.aug import augment_trimap


class TestAugmentAlpha(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.alpha = np.random.uniform(0.0, 255.0, (2, 16, 16, 1)).astype(
            "uint8"
        )

    def test_no_aug(self):
        alpha = augment_alpha(self.alpha, prob=0.0)
        self.assertListEqual(alpha.shape.as_list(), list(self.alpha.shape))

        self.assertEqual(alpha.dtype, "uint8")
        self.assertTupleEqual(tuple(alpha.shape.as_list()), self.alpha.shape)
        self.assertAlmostEqual(self.alpha, alpha)

    def test_aug(self):
        alpha = augment_alpha(self.alpha, prob=1.0)
        self.assertNotAllClose(self.alpha, alpha)


class TestAugmentTrimap(testing.TestCase):
    def setUp(self):
        super().setUp()
        trimap = np.zeros((2, 16, 16, 1), "uint8")
        trimap[:, :4, ...] = 128
        trimap[:, -4:, ...] = 255
        self.trimap = trimap

    def test_no_aug(self):
        trimap = augment_trimap(self.trimap, prob=0.0)
        self.assertAlmostEqual(self.trimap, trimap)

    def test_aug(self):
        trimap = augment_trimap(self.trimap, prob=1.0)
        trimap = backend.convert_to_numpy(trimap)
        self.assertNotAllClose(self.trimap, trimap)
        self.assertSetEqual(set(trimap.ravel()), {0, 128})
