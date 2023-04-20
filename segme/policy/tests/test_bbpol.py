import tensorflow as tf
import unittest
from segme.policy.bbpol import BackbonePolicy, default_policy, global_policy, set_global_policy, policy_scope, \
    serialize, deserialize


class TestBackbonePolicy(unittest.TestCase):
    def test_default(self):
        self.assertEqual(default_policy().name, 'resnet_rs_50-imagenet')

    def test_parse(self):
        p = BackbonePolicy('resnet_rs_50-imagenet')
        self.assertEqual(p.name, 'resnet_rs_50-imagenet')
        self.assertEqual(p.arch_type, 'resnet_rs_50')
        self.assertEqual(p.init_type, 'imagenet')

        self.assertDictEqual(p.get_config(), {'name': 'resnet_rs_50-imagenet'})
        self.assertDictEqual(p.get_config(), BackbonePolicy.from_config(p.get_config()).get_config())

    def test_global(self):
        self.assertIsInstance(global_policy(), BackbonePolicy)
        self.assertEqual(global_policy().name, 'resnet_rs_50-imagenet')

        set_global_policy('swin_tiny_224-none')
        self.assertIsInstance(global_policy(), BackbonePolicy)
        self.assertEqual(global_policy().name, 'swin_tiny_224-none')

        set_global_policy('resnet_rs_50-imagenet')
        self.assertIsInstance(global_policy(), BackbonePolicy)
        self.assertEqual(global_policy().name, 'resnet_rs_50-imagenet')

        set_global_policy(None)
        self.assertIsInstance(global_policy(), BackbonePolicy)
        self.assertEqual(global_policy().name, 'resnet_rs_50-imagenet')

    def test_scope(self):
        self.assertIsInstance(global_policy(), BackbonePolicy)
        self.assertEqual(global_policy().name, 'resnet_rs_50-imagenet')

        with policy_scope('swin_tiny_224-none'):
            self.assertIsInstance(global_policy(), BackbonePolicy)
            self.assertEqual(global_policy().name, 'swin_tiny_224-none')

        self.assertIsInstance(global_policy(), BackbonePolicy)
        self.assertEqual(global_policy().name, 'resnet_rs_50-imagenet')

    def test_serialize(self):
        policy = BackbonePolicy('resnet_rs_50-imagenet')

        config = serialize(policy)
        self.assertDictEqual(config, {
            'module': 'segme.policy.bbpol', 'class_name': 'BackbonePolicy', 'config': {'name': 'resnet_rs_50-imagenet'},
            'registered_name': 'BackbonePolicy'})

        instance = deserialize(config)
        self.assertDictEqual(policy.get_config(), instance.get_config())


if __name__ == '__main__':
    tf.test.main()
