import inspect
import tensorflow as tf
from keras.src import layers
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="SegMe>Common")
class Sequence(layers.Layer):
    def __init__(self, items=None, **kwargs):
        super().__init__(**kwargs)

        items = items or []
        items = items if isinstance(items, (list, tuple)) else [items]

        self.items = []
        self.argspecs = []
        for item in items:
            self.add(item)

    @property
    def supports_masking(self):
        return all([True] + [i.supports_masking for i in self.items])

    def add(self, item):
        if self.built:
            raise ValueError(
                f"Unable to add new layer: {self.name} is already built."
            )

        if not isinstance(item, layers.Layer):
            raise ValueError(
                f"Expected keras.src.layers.Layer instance, got {item}"
            )

        self.items.append(item)
        self.argspecs.append(inspect.getfullargspec(item.call).args)

        idx = len(self.items)  # TODO: remove?
        setattr(self, f"layer_{idx}", item)

    def build(self, input_shape):
        current_shape = input_shape
        for item in self.items:
            if not item.built:
                item.build(current_shape)
            current_shape = item.compute_output_shape(current_shape)

        super().build(input_shape)

    def call(self, inputs, training=False, mask=None):
        outputs = inputs

        if mask is not None:
            try:
                outputs._keras_mask = mask
            except AttributeError:
                # tensor is a C type.
                pass

        for item, argspec in zip(self.items, self.argspecs):
            kwargs = {}
            if "training" in argspec:
                kwargs["training"] = training

            outputs = item(outputs, **kwargs)

        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for item in self.items:
            output_shape = item.compute_output_shape(output_shape)

        return output_shape

    def compute_output_spec(self, input_spec, *args, **kwargs):
        output_spec = input_spec
        for item in self.items:
            output_spec = item.compute_output_spec(output_spec)

        return output_spec

    def get_config(self):
        config = super().get_config()

        items = None
        if self.items:
            items = [layers.serialize(item) for item in self.items]
        config.update({"items": items})

        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        items = config.pop("items", None)
        if items:
            items = [layers.deserialize(item) for item in items]

        return cls(items, **config)
