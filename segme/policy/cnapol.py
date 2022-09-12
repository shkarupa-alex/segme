import contextlib
from keras.utils.generic_utils import deserialize_keras_object, serialize_keras_object
from segme.policy.conv import CONVOLUTIONS
from segme.policy.norm import NORMALIZATIONS
from segme.policy.act import ACTIVATIONS


class ConvNormActPolicy:
    """
    Some popular policies are:
    - conv-bn-relu
    - conv-ln-gelu
    - stdconv-gn-leakyrelu
    - snconv-bn-relu
    - conv-frn-tlu
    """

    def __init__(self, name):
        self._name = name
        if not isinstance(self._name, str):
            raise TypeError(f'Policy name must be a string, got {self._name}')

        if self._name.count('-') != 2:
            raise ValueError('Policy name should cotain 3 parts separated with "-"')

        self._conv_type, self._norm_type, self._act_type = self._name.split('-')
        if self._conv_type not in CONVOLUTIONS:
            raise ValueError(f'Convolution {self._conv_type} not registered')
        if self._norm_type not in NORMALIZATIONS:
            raise ValueError(f'Normalization {self._norm_type} not registered')
        if self._act_type not in ACTIVATIONS:
            raise ValueError(f'Activation {self._act_type} not registered')

    @property
    def name(self):
        return self._name

    @property
    def conv_type(self):
        return self._conv_type

    @property
    def norm_type(self):
        return self._norm_type

    @property
    def act_type(self):
        return self._act_type

    def __repr__(self):
        return f'ConvNormAct policy "{self._name}"'

    def get_config(self):
        return {'name': self._name}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        del custom_objects

        return cls(**config)


_global_policy = None


def default_policy():
    return ConvNormActPolicy('conv-bn-relu')


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
    if isinstance(config, ConvNormActPolicy):
        return config

    if isinstance(config, str):
        return ConvNormActPolicy(config)

    if config is None:
        return default_policy()

    return deserialize_keras_object(
        config,
        module_objects={'ConvNormActPolicy': ConvNormActPolicy},
        custom_objects=custom_objects,
        printable_module_name='conv/norm/act policy')
