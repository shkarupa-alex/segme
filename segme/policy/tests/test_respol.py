import numpy as np
import tensorflow as tf
import unittest
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from segme.policy.respol import ResizePolicy, default_policy, global_policy, set_global_policy, policy_scope, \
    serialize, deserialize


class TestResizePolicy(unittest.TestCase):
    def test_default(self):
        self.assertEqual(default_policy().name, 'inter_linear')

    def test_parse(self):
        p = ResizePolicy('inter_linear')
        self.assertEqual(p.name, 'inter_linear')

        self.assertDictEqual(p.get_config(), {'name': 'inter_linear'})
        self.assertDictEqual(p.get_config(), ResizePolicy.from_config(p.get_config()).get_config())

    def test_global(self):
        self.assertIsInstance(global_policy(), ResizePolicy)
        self.assertEqual(global_policy().name, 'inter_linear')

        set_global_policy('inter_liif')
        self.assertIsInstance(global_policy(), ResizePolicy)
        self.assertEqual(global_policy().name, 'inter_liif')

        set_global_policy('inter_linear')
        self.assertIsInstance(global_policy(), ResizePolicy)
        self.assertEqual(global_policy().name, 'inter_linear')

        set_global_policy(None)
        self.assertIsInstance(global_policy(), ResizePolicy)
        self.assertEqual(global_policy().name, 'inter_linear')

    def test_scope(self):
        self.assertIsInstance(global_policy(), ResizePolicy)
        self.assertEqual(global_policy().name, 'inter_linear')

        with policy_scope('inter_liif'):
            self.assertIsInstance(global_policy(), ResizePolicy)
            self.assertEqual(global_policy().name, 'inter_liif')

        self.assertIsInstance(global_policy(), ResizePolicy)
        self.assertEqual(global_policy().name, 'inter_linear')

    def test_serialize(self):
        policy = ResizePolicy('inter_linear')

        config = serialize(policy)
        self.assertDictEqual(config, {'class_name': 'ResizePolicy', 'config': {'name': 'inter_linear'}})

        instance = deserialize(config)
        self.assertDictEqual(policy.get_config(), instance.get_config())


if __name__ == '__main__':
    tf.test.main()
