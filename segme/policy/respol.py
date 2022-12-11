import contextlib
from keras.utils import deserialize_keras_object, serialize_keras_object
from segme.policy.resize import RESIZERS


class ResizePolicy:
    """
    Some popular policies are:
    - inter_linear
    - inter_liif
    """

    def __init__(self, name):
        self._name = name
        if not isinstance(self._name, str):
            raise TypeError(f'Policy name must be a string, got {self._name}')

        if self._name not in RESIZERS:
            raise ValueError(f'Resize method {self._name} not registered')

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f'Resize policy "{self._name}"'

    def get_config(self):
        return {'name': self._name}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        del custom_objects

        return cls(**config)


_global_policy = None


def default_policy():
    return ResizePolicy('inter_linear')


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
    if isinstance(config, ResizePolicy):
        return config

    if isinstance(config, str):
        return ResizePolicy(config)

    if config is None:
        return default_policy()

    return deserialize_keras_object(
        config,
        module_objects={'ResizePolicy': ResizePolicy},
        custom_objects=custom_objects,
        printable_module_name='resize policy')
