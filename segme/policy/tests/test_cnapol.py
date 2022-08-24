import numpy as np
import tensorflow as tf
import unittest
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from segme.policy.cnapol import ConvNormActPolicy, default_policy, global_policy, set_global_policy, policy_scope, \
    serialize, deserialize


class TestConvNormActPolicy(unittest.TestCase):
    def test_default(self):
        self.assertEqual(default_policy().name, 'conv_bn_relu')

    def test_parse(self):
        p = ConvNormActPolicy('conv_bn_relu')
        self.assertEqual(p.name, 'conv_bn_relu')
        self.assertEqual(p.conv_type, 'conv')
        self.assertEqual(p.norm_type, 'bn')
        self.assertEqual(p.act_type, 'relu')

        self.assertDictEqual(p.get_config(), {'name': 'conv_bn_relu'})
        self.assertDictEqual(p.get_config(), ConvNormActPolicy.from_config(p.get_config()).get_config())

    def test_global(self):
        self.assertIsInstance(global_policy(), ConvNormActPolicy)
        self.assertEqual(global_policy().name, 'conv_bn_relu')

        set_global_policy('stdconv_gn_leakyrelu')
        self.assertIsInstance(global_policy(), ConvNormActPolicy)
        self.assertEqual(global_policy().name, 'stdconv_gn_leakyrelu')

        set_global_policy('conv_bn_relu')
        self.assertIsInstance(global_policy(), ConvNormActPolicy)
        self.assertEqual(global_policy().name, 'conv_bn_relu')

        set_global_policy(None)
        self.assertIsInstance(global_policy(), ConvNormActPolicy)
        self.assertEqual(global_policy().name, 'conv_bn_relu')

    def test_scope(self):
        self.assertIsInstance(global_policy(), ConvNormActPolicy)
        self.assertEqual(global_policy().name, 'conv_bn_relu')

        with policy_scope('stdconv_gn_leakyrelu'):
            self.assertIsInstance(global_policy(), ConvNormActPolicy)
            self.assertEqual(global_policy().name, 'stdconv_gn_leakyrelu')

        self.assertIsInstance(global_policy(), ConvNormActPolicy)
        self.assertEqual(global_policy().name, 'conv_bn_relu')

    def test_serialize(self):
        policy = ConvNormActPolicy('conv_bn_relu')

        config = serialize(policy)
        self.assertDictEqual(config, {'class_name': 'ConvNormActPolicy', 'config': {'name': 'conv_bn_relu'}})

        instance = deserialize(config)
        self.assertDictEqual(policy.get_config(), instance.get_config())


if __name__ == '__main__':
    tf.test.main()
