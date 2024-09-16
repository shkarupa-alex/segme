import unittest

from keras.src import layers

from segme.policy.backbone.backbone import BACKBONES


class TestBackboneRegistry(unittest.TestCase):
    def test_filled(self):
        self.assertIn("efficientnet_v2_small", BACKBONES)
        self.assertIn("swin_v2_tiny_256", BACKBONES)
        self.assertIn("resnet_rs_50", BACKBONES)

    def test_input_tensor(self):
        image = layers.Input(name="image", shape=(None, None, 2), dtype="uint8")
        mask = layers.Input(name="mask", shape=(None, None, 1), dtype="uint8")
        combo = layers.concatenate([image, mask], axis=-1, name="concat")

        bbone = BACKBONES.new(
            "efficientnet_v2_small", None, None, input_tensor=combo
        )
        self.assertEqual(len(bbone.inputs), 2)
        self.assertEqual(bbone.inputs[0], image)
        self.assertEqual(bbone.inputs[1], mask)

    def test_input_tensor_restored(self):
        image = layers.Input(name="image", shape=(None, None, 2), dtype="uint8")
        mask = layers.Input(name="mask", shape=(None, None, 1), dtype="uint8")
        combo = layers.concatenate([image, mask], axis=-1, name="concat")

        bbone = BACKBONES.new("resnet_rs_50_s8", None, None, input_tensor=combo)
        self.assertEqual(len(bbone.inputs), 2)
        self.assertEqual(bbone.inputs[0].shape, image.shape)
        self.assertEqual(bbone.inputs[0].dtype, image.dtype)
        self.assertEqual(bbone.inputs[1].shape, mask.shape)
        self.assertEqual(bbone.inputs[1].dtype, mask.dtype)
