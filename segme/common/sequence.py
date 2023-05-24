import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_inspect import getfullargspec
from keras.src.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='SegMe>Common')
class Sequenсe(layers.Layer):
    def __init__(self, items=None, **kwargs):
        kwargs['autocast'] = False
        super().__init__(**kwargs)

        items = items or []
        items = items if isinstance(items, (list, tuple)) else [items]

        self.items = []
        self.argspecs = []
        for item in items:
            self.add(item)

    def add(self, item):
        if self.built:
            raise ValueError(f'Unable to add new layer: {self.name} is already built.')

        if not isinstance(item, layers.Layer):
            raise ValueError(f'Expected keras.layers.Layer instance, got {item}')

        self.items.append(item)
        self.argspecs.append(getfullargspec(item.call).args)

        idx = len(self.items)
        setattr(self, f'layer_{idx}', item)

    def call(self, inputs, training=None, mask=None):
        outputs = inputs

        for item, argspec in zip(self.items, self.argspecs):
            kwargs = {}
            if 'mask' in argspec:
                kwargs['mask'] = mask
            if 'training' in argspec:
                kwargs['training'] = training

            outputs = item(outputs, **kwargs)

            if 1 == len(tf.nest.flatten(outputs)):
                mask = getattr(outputs, '_keras_mask', None)

        return outputs

    def compute_mask(self, inputs, mask=None):
        outputs = self.call(inputs, mask=mask)

        if 1 == len(tf.nest.flatten(outputs)):
            return getattr(outputs, '_keras_mask', None)

        return None

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for item in self.items:
            output_shape = item.compute_output_shape(output_shape)

        return output_shape

    def compute_output_signature(self, input_signature):
        output_signature = input_signature
        for item in self.items:
            output_signature = item.compute_output_signature(output_signature)

        return output_signature

    def get_config(self):
        config = super().get_config()

        items = None
        if self.items:
            items = [layers.serialize(item) for item in self.items]
        config.update({'items': items})

        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        items = config.pop('items', None)
        if items:
            items = [layers.deserialize(item) for item in items]

        return cls(items, **config)
