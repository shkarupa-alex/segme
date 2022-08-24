import copy
import inspect
from keras import layers
from keras.utils.generic_utils import get_custom_objects_by_name


class Registry:
    def __init__(self):
        self.__registry = {}

    def register(self, key):
        def wrapper(value):
            self._validate(key, value)
            self.__registry[key] = value

            return value

        return wrapper

    def _validate(self, key, value):
        if not isinstance(key, str):
            raise ValueError(f'Expected key be a string, got {key}')

        if key in self.__registry:
            old = self.__registry[key]
            raise ValueError(f'Key "{key}" already registered')

    def __getitem__(self, key):
        return self.__registry[key]

    def __contains__(self, key):
        return key in self.__registry

    def __len__(self):
        return len(self.__registry)

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.__registry})'


class LayerRegistry(Registry):
    def _validate(self, key, value):
        super()._validate(key, value)

        if isinstance(value, dict):
            if 'class_name' not in value:
                raise ValueError(f'Expected serialized layer config, got {value} for key "{key}"')
        elif inspect.isclass(value):
            if not issubclass(value, layers.Layer):
                raise ValueError(f'Expected keras.layers.Layer subclass, got {value} for key "{key}"')
        else:
            raise ValueError(f'Only serialized layer config and keras.layers.Layer subclass '
                             f'can be registered, got {value} for key "{key}"')

    def _kwargs(self, cls):
        if layers.Layer == cls:
            return {'trainable', 'name', 'dtype', 'dynamic', 'activity_regularizer', 'autocast', 'weights'}

        signature = inspect.signature(cls.__init__)
        kwargs = {k for k, _ in signature.parameters.items() if k not in {'self', 'args', 'kwargs'}}

        if 1 == len(cls.__bases__) and 'kwargs' in signature.parameters.keys():
            kwargs |= self._kwargs(cls.__bases__[0])

        return kwargs

    def new(self, key, *args, **kwargs):
        value = self[key]

        is_class = inspect.isclass(value)

        if is_class:
            cls = value
        elif isinstance(value, dict):
            cls = layers.get_builtin_layer(value['class_name']) or get_custom_objects_by_name(value['class_name'])
            if cls is None:
                raise ValueError(f'Can\'t find class for {value}')
        else:
            raise ValueError(f'Only serialized layer config and keras.layers.Layer subclass '
                             f'can be registered, got {value} for key "{key}"')

        signature = inspect.signature(cls.__init__)
        posargs = [k for k, _ in signature.parameters.items() if k not in {'self', 'args', 'kwargs'}]
        posargs = {k: v for k, v in zip(posargs, args)}

        crossargs = set(posargs.keys()) & set(kwargs.keys())
        if crossargs:
            raise TypeError(f'Got multiple values for arguments: {crossargs}')

        dropkwargs = set(kwargs.keys()) - self._kwargs(cls)
        kwargs = {k: v for k, v in kwargs.items() if k not in dropkwargs}
        kwargs.update(posargs)

        if is_class:
            return value(**kwargs)

        config = copy.deepcopy(value)
        config['config'] = config.get('config', {})
        config['config'].update(kwargs)

        return layers.deserialize(config)
