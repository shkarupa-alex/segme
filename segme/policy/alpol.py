import contextlib

from keras.src.saving.serialization_lib import deserialize_keras_object
from keras.src.saving.serialization_lib import serialize_keras_object

from segme.policy.align.align import ALIGNERS


class AlignPolicy:
    """
    Some popular policies are:
    - linear
    - deform
    """

    def __init__(self, name):
        self._name = name
        if not isinstance(self._name, str):
            raise TypeError(f"Policy name must be a string, got {self._name}")

        if self._name not in ALIGNERS:
            raise ValueError(f"Align method {self._name} not registered")

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f'Align policy "{self._name}"'

    def get_config(self):
        return {"name": self._name}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        del custom_objects

        return cls(**config)


_global_policy = None


def default_policy():
    return AlignPolicy("linear")


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
    if isinstance(config, AlignPolicy):
        return config

    if isinstance(config, str):
        return AlignPolicy(config)

    if config is None:
        return default_policy()

    return deserialize_keras_object(
        config,
        module_objects={"AlignPolicy": AlignPolicy},
        custom_objects=custom_objects,
        printable_module_name="align policy",
    )
