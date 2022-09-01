import numpy as np
import tensorflow as tf
import unittest
from segme.policy.backbone.backbone import BACKBONES


class TestBackboneRegistry(unittest.TestCase):
    def test_filled(self):
        self.assertIn('efficientnet_v2_small', BACKBONES)
        self.assertIn('swin_v2_tiny_256', BACKBONES)
        self.assertIn('resnet_rs_50', BACKBONES)


if __name__ == '__main__':
    tf.test.main()
