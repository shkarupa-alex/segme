import unittest

from segme.policy.alpol import AlignPolicy
from segme.policy.alpol import default_policy
from segme.policy.alpol import deserialize
from segme.policy.alpol import global_policy
from segme.policy.alpol import policy_scope
from segme.policy.alpol import serialize
from segme.policy.alpol import set_global_policy


class TestAlignPolicy(unittest.TestCase):
    def test_default(self):
        self.assertEqual(default_policy().name, "linear")

    def test_parse(self):
        p = AlignPolicy("linear")
        self.assertEqual(p.name, "linear")

        self.assertDictEqual(p.get_config(), {"name": "linear"})
        self.assertDictEqual(
            p.get_config(), AlignPolicy.from_config(p.get_config()).get_config()
        )

    def test_global(self):
        self.assertIsInstance(global_policy(), AlignPolicy)
        self.assertEqual(global_policy().name, "linear")

        set_global_policy("deconv3")
        self.assertIsInstance(global_policy(), AlignPolicy)
        self.assertEqual(global_policy().name, "deconv3")

        set_global_policy("linear")
        self.assertIsInstance(global_policy(), AlignPolicy)
        self.assertEqual(global_policy().name, "linear")

        set_global_policy(None)
        self.assertIsInstance(global_policy(), AlignPolicy)
        self.assertEqual(global_policy().name, "linear")

    def test_scope(self):
        self.assertIsInstance(global_policy(), AlignPolicy)
        self.assertEqual(global_policy().name, "linear")

        with policy_scope("deconv4"):
            self.assertIsInstance(global_policy(), AlignPolicy)
            self.assertEqual(global_policy().name, "deconv4")

        self.assertIsInstance(global_policy(), AlignPolicy)
        self.assertEqual(global_policy().name, "linear")

    def test_serialize(self):
        policy = AlignPolicy("linear")

        config = serialize(policy)
        self.assertDictEqual(
            config,
            {
                "module": "segme.policy.alpol",
                "class_name": "AlignPolicy",
                "config": {"name": "linear"},
                "registered_name": "AlignPolicy",
            },
        )

        instance = deserialize(config)
        self.assertDictEqual(policy.get_config(), instance.get_config())
