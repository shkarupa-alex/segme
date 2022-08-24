import unittest
from keras import layers
from segme.registry.registry import Registry, LayerRegistry
from segme.common.pad import SymmetricPadding  # requred to be registered as serializable


class TestRegistry(unittest.TestCase):
    def test_independent(self):
        r1, r2 = Registry(), Registry()
        v1 = r1.register('k1')(1)
        v2 = r1.register('k2')(2)
        v3 = r2.register('k1')(3)

        self.assertIn('k1', r1)
        self.assertEqual(r1['k1'], v1)
        self.assertEqual(v1, 1)

        self.assertIn('k2', r1)
        self.assertEqual(r1['k2'], v2)
        self.assertEqual(v2, 2)

        self.assertIn('k1', r2)
        self.assertEqual(r2['k1'], v3)
        self.assertEqual(v3, 3)

    def test_magic(self):
        r = Registry()
        r.register('k1')(1)
        r.register('k2')(2)

        self.assertEqual(len(r), 2)
        self.assertEqual(str(r), 'Registry ({\'k1\': 1, \'k2\': 2})')


class TestLayerRegistry(unittest.TestCase):
    def test_config(self):
        r = LayerRegistry()

        r.register('bn')({'class_name': 'BatchNormalization'})
        instance = r.new('bn')
        self.assertIsInstance(instance, layers.BatchNormalization)
        self.assertEqual(instance.get_config()['center'], True)

        r.register('sp_custom')({'class_name': 'SegMe>Common>SymmetricPadding'})
        instance = r.new('sp_custom')
        self.assertIsInstance(instance, SymmetricPadding)

        r.register('bn_conf')({'class_name': 'BatchNormalization', 'config': {'center': False}})
        instance = r.new('bn_conf')
        self.assertIsInstance(instance, layers.BatchNormalization)
        self.assertEqual(instance.get_config()['center'], False)

        r.register('bn_arg')({'class_name': 'BatchNormalization'})
        instance = r.new('bn_arg', 0)
        self.assertIsInstance(instance, layers.BatchNormalization)
        self.assertEqual(instance.get_config()['axis'], 0)

        r.register('bn_kwarg')({'class_name': 'BatchNormalization'})
        instance = r.new('bn_kwarg', center=False)
        self.assertIsInstance(instance, layers.BatchNormalization)
        self.assertEqual(instance.get_config()['center'], False)

        r.register('conv_arg_req')({'class_name': 'Conv2D'})
        instance = r.new('conv_arg_req', 16, 3)
        self.assertIsInstance(instance, layers.Conv2D)
        self.assertEqual(instance.get_config()['filters'], 16)
        self.assertTupleEqual(instance.get_config()['kernel_size'], (3, 3))

        r.register('conv_arg_kwarg')({'class_name': 'Conv2D'})
        instance = r.new('conv_arg_kwarg', 16, 3, 2, activation='relu')
        self.assertIsInstance(instance, layers.Conv2D)
        self.assertEqual(instance.get_config()['filters'], 16)
        self.assertTupleEqual(instance.get_config()['kernel_size'], (3, 3))
        self.assertTupleEqual(instance.get_config()['strides'], (2, 2))
        self.assertEqual(instance.get_config()['activation'], 'relu')

        r.register('conv_arg_kwarg_same')({'class_name': 'Conv2D'})
        with self.assertRaisesRegex(TypeError, 'Got multiple values'):
            r.new('conv_arg_kwarg_same', 16, 3, 2, strides=2)

    def test_class(self):
        r = LayerRegistry()

        r.register('bn')(layers.BatchNormalization)
        instance = r.new('bn')
        self.assertIsInstance(instance, layers.BatchNormalization)
        self.assertEqual(instance.get_config()['center'], True)

        r.register('bn_arg')(layers.BatchNormalization)
        instance = r.new('bn_arg', 0)
        self.assertIsInstance(instance, layers.BatchNormalization)
        self.assertEqual(instance.get_config()['axis'], 0)

        r.register('bn_kwarg')(layers.BatchNormalization)
        instance = r.new('bn_kwarg', center=False)
        self.assertIsInstance(instance, layers.BatchNormalization)
        self.assertEqual(instance.get_config()['center'], False)

        r.register('conv_arg_req')(layers.Conv2D)
        instance = r.new('conv_arg_req', 16, 3)
        self.assertIsInstance(instance, layers.Conv2D)
        self.assertEqual(instance.get_config()['filters'], 16)
        self.assertTupleEqual(instance.get_config()['kernel_size'], (3, 3))

        r.register('conv_arg_kwarg')(layers.Conv2D)
        instance = r.new('conv_arg_kwarg', 16, 3, 2, activation='relu')
        self.assertIsInstance(instance, layers.Conv2D)
        self.assertEqual(instance.get_config()['filters'], 16)
        self.assertTupleEqual(instance.get_config()['kernel_size'], (3, 3))
        self.assertTupleEqual(instance.get_config()['strides'], (2, 2))
        self.assertEqual(instance.get_config()['activation'], 'relu')

        r.register('conv_arg_kwarg_same')(layers.Conv2D)
        with self.assertRaisesRegex(TypeError, 'Got multiple values'):
            r.new('conv_arg_kwarg_same', 16, 3, 2, strides=2)
