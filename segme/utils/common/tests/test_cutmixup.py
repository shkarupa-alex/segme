import tensorflow as tf

from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff
from segme.utils.common.cutmixup import _cut_mix
from segme.utils.common.cutmixup import _mix_up
from segme.utils.common.cutmixup import cut_mix_up


class TestCutMixUp(tf.test.TestCase):
    def test_apply(self):
        images = tf.random.uniform([8, 32, 32, 3])
        labels = tf.random.uniform([8, 1], maxval=4, dtype="int32")
        weights = tf.random.uniform([8, 1], dtype="float32")

        images_, labels_, weights_ = cut_mix_up(images, labels, weights, 4)

        self.assertShapeEqual(images_, images)
        self.assertDTypeEqual(images_, images.dtype)

        self.assertShapeEqual(labels_, tf.repeat(labels, 4, axis=-1))
        self.assertDTypeEqual(labels_, tf.float32)

        self.assertShapeEqual(weights_, weights)
        self.assertDTypeEqual(weights_, weights.dtype)


class TestCutMix(tf.test.TestCase):
    def test_ref(self):
        images, expected_images = aug_samples("cutmix")
        labels = tf.constant([0, 1, 2, 3, 4, 5], "int32")
        expected_labels = tf.constant(
            [
                [0.995, 0.0, 0.0, 0.0, 0.0, 0.005],
                [0.0, 0.985, 0.0, 0.0, 0.015, 0.0],
                [0.0, 0.0, 0.969, 0.031, 0.0, 0.0],
                [0.0, 0.0, 0.051, 0.949, 0.0, 0.0],
                [0.0, 0.077, 0.0, 0.0, 0.923, 0.0],
                [0.107, 0.0, 0.0, 0.0, 0.0, 0.893],
            ],
            "float32",
        )
        weights = tf.constant([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "float32")
        expected_weights = tf.constant(
            [0.003, 0.105, 0.203, 0.295, 0.377, 0.446], "float32"
        )

        augmented = _cut_mix(
            images,
            labels,
            weights,
            [5, 4, 3, 2, 1, 0],
            tf.cast(
                [[0, 0], [32, 32], [64, 48], [96, 64], [128, 80], [160, 96]],
                "float32",
            ),
            tf.cast(
                [
                    [32, 32],
                    [48, 64],
                    [64, 96],
                    [80, 128],
                    [96, 160],
                    [112, 192],
                ],
                "float32",
            ),
            6,
        )

        image_diff = max_diff(expected_images, augmented[0])
        self.assertLessEqual(image_diff, 1e-5)

        label_diff = max_diff(expected_labels, augmented[1])
        self.assertLessEqual(label_diff, 1e-3)

        weight_diff = max_diff(expected_weights, augmented[2])
        self.assertLessEqual(weight_diff, 1e-3)

    def test_float(self):
        images, expected_images = aug_samples("cutmix", "float32")
        labels = tf.constant(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            "float32",
        )
        expected_labels = tf.constant(
            [
                [0.995, 0.0, 0.0, 0.0, 0.0, 0.005],
                [0.0, 0.985, 0.0, 0.0, 0.015, 0.0],
                [0.0, 0.0, 0.969, 0.031, 0.0, 0.0],
                [0.0, 0.0, 0.051, 0.949, 0.0, 0.0],
                [0.0, 0.077, 0.0, 0.0, 0.923, 0.0],
                [0.107, 0.0, 0.0, 0.0, 0.0, 0.893],
            ],
            "float32",
        )
        weights = tf.constant([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "float32")
        expected_weights = tf.constant(
            [0.003, 0.105, 0.203, 0.295, 0.377, 0.446], "float32"
        )

        augmented = _cut_mix(
            images,
            labels,
            weights,
            [5, 4, 3, 2, 1, 0],
            tf.cast(
                [[0, 0], [32, 32], [64, 48], [96, 64], [128, 80], [160, 96]],
                "float32",
            ),
            tf.cast(
                [
                    [32, 32],
                    [48, 64],
                    [64, 96],
                    [80, 128],
                    [96, 160],
                    [112, 192],
                ],
                "float32",
            ),
            6,
        )

        image_diff = max_diff(expected_images, augmented[0])
        self.assertLessEqual(image_diff, 1 / 255)

        label_diff = max_diff(expected_labels, augmented[1])
        self.assertLessEqual(label_diff, 1e-3)

        weight_diff = max_diff(expected_weights, augmented[2])
        self.assertLessEqual(weight_diff, 1e-3)


class TestMixUp(tf.test.TestCase):
    def test_ref(self):
        images, expected_images = aug_samples("mixup")
        labels = tf.constant([0, 1, 2, 3, 4, 5], "int32")
        expected_labels = tf.constant(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0.8, 0, 0, 0.2, 0],
                [0, 0, 0.6, 0.4, 0, 0],
                [0, 0, 0.6, 0.4, 0, 0],
                [0, 0.8, 0, 0, 0.2, 0],
                [1, 0, 0, 0, 0, 0],
            ],
            "float32",
        )
        weights = tf.constant([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "float32")
        expected_weights = tf.constant(
            [0.0, 0.16, 0.24, 0.24, 0.16, 0.0], "float32"
        )

        augmented = _mix_up(
            images,
            labels,
            weights,
            [5, 4, 3, 2, 1, 0],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            6,
        )

        image_diff = max_diff(expected_images, augmented[0])
        self.assertLessEqual(image_diff, 1e-5)

        label_diff = max_diff(expected_labels, augmented[1])
        self.assertLessEqual(label_diff, 1e-5)

        weight_diff = max_diff(expected_weights, augmented[2])
        self.assertLessEqual(weight_diff, 1e-5)

    def test_float(self):
        images, expected_images = aug_samples("mixup", "float32")
        labels = tf.constant(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            "float32",
        )
        expected_labels = tf.constant(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0.8, 0, 0, 0.2, 0],
                [0, 0, 0.6, 0.4, 0, 0],
                [0, 0, 0.6, 0.4, 0, 0],
                [0, 0.8, 0, 0, 0.2, 0],
                [1, 0, 0, 0, 0, 0],
            ],
            "float32",
        )
        weights = tf.constant([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "float32")
        expected_weights = tf.constant(
            [0.0, 0.16, 0.24, 0.24, 0.16, 0.0], "float32"
        )

        augmented = _mix_up(
            images,
            labels,
            weights,
            [5, 4, 3, 2, 1, 0],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            6,
        )

        image_diff = max_diff(expected_images, augmented[0])
        self.assertLessEqual(image_diff, 1 / 255)

        label_diff = max_diff(expected_labels, augmented[1])
        self.assertLessEqual(label_diff, 1e-5)

        weight_diff = max_diff(expected_weights, augmented[2])
        self.assertLessEqual(weight_diff, 1e-5)
