import functools
from keras import layers
from segme.policy import alpol, bbpol, cnapol


def preserve_current_layer_policies(cls):
    if not issubclass(cls, layers.Layer):
        raise ValueError(f'Can preserve policies only in `Layer` subclass. Got {cls}')

    setattr(cls, '__init__', _init_decorator(getattr(cls, '__init__')))
    setattr(cls, 'build', _build_decorator(getattr(cls, 'build')))
    setattr(cls, '__call__', _call_decorator(getattr(cls, '__call__')))
    setattr(cls, 'get_config', _get_config_decorator(getattr(cls, 'get_config')))

    return cls


def _init_decorator(fn):
    @functools.wraps(fn)
    def decorate(self, *args, **kwargs):
        self.__alpol = alpol.deserialize(kwargs.pop('__alpol', alpol.global_policy()))
        self.__bbpol = bbpol.deserialize(kwargs.pop('__bbpol', bbpol.global_policy()))
        self.__cnapol = cnapol.deserialize(kwargs.pop('__cnapol', cnapol.global_policy()))

        with alpol.policy_scope(self.__alpol), bbpol.policy_scope(self.__bbpol), cnapol.policy_scope(self.__cnapol):
            fn(self, *args, **kwargs)

    return decorate


def _build_decorator(fn):
    @functools.wraps(fn)
    def decorate(self, *args, **kwargs):
        with alpol.policy_scope(self.__alpol), bbpol.policy_scope(self.__bbpol), cnapol.policy_scope(self.__cnapol):
            fn(self, *args, **kwargs)

    return decorate


def _call_decorator(fn):
    @functools.wraps(fn)
    def decorate(self, *args, **kwargs):
        with alpol.policy_scope(self.__alpol), bbpol.policy_scope(self.__bbpol), cnapol.policy_scope(self.__cnapol):
            return fn(self, *args, **kwargs)

    return decorate


def _get_config_decorator(fn):
    @functools.wraps(fn)
    def decorate(self, *args, **kwargs):
        config = fn(self, *args, **kwargs)
        config.update({
            '__alpol': alpol.serialize(self.__alpol),
            '__bbpol': bbpol.serialize(self.__bbpol),
            '__cnapol': cnapol.serialize(self.__cnapol)
        })

        return config

    return decorate
