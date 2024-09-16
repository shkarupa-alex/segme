import unittest

from segme.policy.cnapol import ConvNormActPolicy
from segme.policy.cnapol import default_policy
from segme.policy.cnapol import deserialize
from segme.policy.cnapol import global_policy
from segme.policy.cnapol import policy_scope
from segme.policy.cnapol import serialize
from segme.policy.cnapol import set_global_policy


class TestConvNormActPolicy(unittest.TestCase):
    def test_default(self):
        self.assertEqual(default_policy().name, "conv-bn-relu")

    def test_parse(self):
        p = ConvNormActPolicy("conv-bn-relu")
        self.assertEqual(p.name, "conv-bn-relu")
        self.assertEqual(p.conv_type, "conv")
        self.assertEqual(p.norm_type, "bn")
        self.assertEqual(p.act_type, "relu")

        self.assertDictEqual(p.get_config(), {"name": "conv-bn-relu"})
        self.assertDictEqual(
            p.get_config(),
            ConvNormActPolicy.from_config(p.get_config()).get_config(),
        )

    def test_global(self):
        self.assertIsInstance(global_policy(), ConvNormActPolicy)
        self.assertEqual(global_policy().name, "conv-bn-relu")

        set_global_policy("stdconv-gn-leakyrelu")
        self.assertIsInstance(global_policy(), ConvNormActPolicy)
        self.assertEqual(global_policy().name, "stdconv-gn-leakyrelu")

        set_global_policy("conv-bn-relu")
        self.assertIsInstance(global_policy(), ConvNormActPolicy)
        self.assertEqual(global_policy().name, "conv-bn-relu")

        set_global_policy(None)
        self.assertIsInstance(global_policy(), ConvNormActPolicy)
        self.assertEqual(global_policy().name, "conv-bn-relu")

    def test_scope(self):
        self.assertIsInstance(global_policy(), ConvNormActPolicy)
        self.assertEqual(global_policy().name, "conv-bn-relu")

        with policy_scope("stdconv-gn-leakyrelu"):
            self.assertIsInstance(global_policy(), ConvNormActPolicy)
            self.assertEqual(global_policy().name, "stdconv-gn-leakyrelu")

        self.assertIsInstance(global_policy(), ConvNormActPolicy)
        self.assertEqual(global_policy().name, "conv-bn-relu")

    def test_serialize(self):
        policy = ConvNormActPolicy("conv-bn-relu")

        config = serialize(policy)
        self.assertDictEqual(
            config,
            {
                "module": "segme.policy.cnapol",
                "class_name": "ConvNormActPolicy",
                "config": {"name": "conv-bn-relu"},
                "registered_name": "ConvNormActPolicy",
            },
        )

        instance = deserialize(config)
        self.assertDictEqual(policy.get_config(), instance.get_config())
