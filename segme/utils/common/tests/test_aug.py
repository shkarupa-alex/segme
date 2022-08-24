import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from segme.utils.common.aug import stateless_random_rotate_90, random_color_mix, random_channel_shuffle, \
    augment_onthefly


@test_util.run_all_in_graph_and_eager_modes
class TestStatelessRandomRotate90(tf.test.TestCase):
    def test_aug(self):
        image = np.random.uniform(0., 255., (4, 16, 16, 3)).astype('uint8')
        expected = np.array([
            image[0],
            cv2.flip(cv2.transpose(image[1]), flipCode=1),
            cv2.flip(cv2.transpose(image[2]), flipCode=0),
            image[3]])

        rotated = stateless_random_rotate_90(image, [0, 1024])
        rotated = self.evaluate(rotated)

        self.assertAllEqual(rotated, expected)


@test_util.run_all_in_graph_and_eager_modes
class TestRandomColorMix(tf.test.TestCase):
    def test_no_aug(self):
        image = np.random.uniform(0., 255., (4, 16, 16, 3)).astype('uint8')
        mixed = random_color_mix(image, 0.)
        mixed = self.evaluate(mixed)

        self.assertAllEqual(mixed, image)

    def test_aug(self):
        image = np.random.uniform(0., 255., (4, 16, 16, 3)).astype('uint8')
        mixed = random_color_mix(image, 1.)
        mixed = self.evaluate(mixed)

        self.assertNotAllEqual(mixed, image)


@test_util.run_all_in_graph_and_eager_modes
class TestRandomChannelShuffle(tf.test.TestCase):
    def test_aug(self):
        expected = {(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)}

        images = np.tile([[[[0, 1, 2]]]], [25, 1, 1, 1])
        shuffled = random_channel_shuffle(images)
        shuffled = self.evaluate(shuffled)
        result = set(tuple(u) for u in np.unique(shuffled[:, 0, 0, :], axis=0))

        self.assertSetEqual(expected, result)


@test_util.run_all_in_graph_and_eager_modes
class TestAugmentOnTheFly(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.image = np.random.uniform(0., 255., (8, 16, 16, 3)).astype('uint8')
        self.masks = [np.random.uniform(0., 255., (8, 16, 16, 1)).astype('uint8'),
                      np.random.uniform(0., 255., (8, 16, 16, 1)).astype('uint8')]

    def test_no_aug(self):
        image, masks = augment_onthefly(
            self.image, self.masks, hflip_prob=0., vflip_prob=0., rotate_prob=0., brightness_prob=0., contrast_prob=0.,
            hue_prob=0., saturation_prob=0., mix_prob=0., shuffle_prob=0.)

        image = self.evaluate(image)
        self.assertAllEqual(self.image, image)

        for i in range(len(self.masks)):
            mask = self.evaluate(masks[i])
            self.assertAllEqual(self.masks[i], mask)

    def test_aug(self):
        kwargs = {'hflip_prob': 0., 'vflip_prob': 0., 'rotate_prob': 0., 'brightness_prob': 0., 'contrast_prob': 0.,
                  'hue_prob': 0., 'saturation_prob': 0., 'mix_prob': 0., 'shuffle_prob': 0.}
        max_prob = {'hflip_prob': 0.5, 'vflip_prob': 0.5, 'rotate_prob': 0.66}

        for aug in kwargs.keys():
            kwargs_ = dict(kwargs)
            kwargs_[aug] = max_prob.get(aug, 1.)

            image, masks = augment_onthefly(self.image, self.masks, **kwargs_)

            image = self.evaluate(image)
            self.assertNotAllEqual(self.image, image)

            if aug in {'hflip_prob', 'vflip_prob', 'rotate_prob'}:
                for i in range(len(self.masks)):
                    mask = self.evaluate(masks[i])
                    self.assertNotAllEqual(self.masks[i], mask)


if __name__ == '__main__':
    tf.test.main()
