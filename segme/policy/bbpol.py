import contextlib
import tensorflow as tf
from keras.utils.generic_utils import deserialize_keras_object, serialize_keras_object
from segme.policy.backbone.backbone import BACKBONES


class BackbonePolicy:
    """
    Some popular policies are:
    - resnet_rs_50-imagenet
    - swin_v2_tiny_256-imagenet
    - mobilenet_v3_large-imagenet
    """

    def __init__(self, name):
        self._name = name
        if not isinstance(self._name, str):
            raise TypeError(f'Policy name must be a string, got {self._name}')

        if not self._name.count('-'):
            raise ValueError('Policy name should cotain 2 parts separated with "-"')

        name_parts = self._name.split('-')
        self._arch_type, self._init_type = name_parts[0], '-'.join(name_parts[1:])
        if self._arch_type not in BACKBONES:
            raise ValueError(f'Backbone {self._arch_type} not registered')
        if not (self._init_type in {'imagenet', 'none'} or tf.io.gfile.exists(self._init_type)):
            raise ValueError(f'Unknown init type {self._init_type}')
        if 'none' == self._init_type:
            self._init_type = None

    @property
    def name(self):
        return self._name

    @property
    def arch_type(self):
        return self._arch_type

    @property
    def init_type(self):
        return self._init_type

    def __repr__(self):
        return f'Backbone policy "{self._name}"'

    def get_config(self):
        return {'name': self._name}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        del custom_objects

        return cls(**config)


_global_policy = None


def default_policy():
    return BackbonePolicy('resnet_rs_50-imagenet')


def global_policy():
    global _global_policy

    if _global_policy is None:
        set_global_policy(default_policy())

    return _global_policy


def set_global_policy(policy):
    global _global_policy
    _global_policy = deserialize(policy)


@contextlib.contextmanager
def policy_scope(policy):
    old_policy = global_policy()
    try:
        set_global_policy(policy)
        yield
    finally:
        set_global_policy(old_policy)


def serialize(policy):
    return serialize_keras_object(policy)


def deserialize(config, custom_objects=None):
    if isinstance(config, BackbonePolicy):
        return config

    if isinstance(config, str):
        return BackbonePolicy(config)

    if config is None:
        return default_policy()

    return deserialize_keras_object(
        config,
        module_objects={'BackbonePolicy': BackbonePolicy},
        custom_objects=custom_objects,
        printable_module_name='backbone policy')
