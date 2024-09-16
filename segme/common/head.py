from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="SegMe>Common")
class HeadProjection(layers.Layer):
    def __init__(
        self,
        classes,
        kernel_size=1,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)
        self.classes = classes
        self.kernel_size = kernel_size
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        self.proj = layers.Conv2D(
            self.classes,
            self.kernel_size,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            name="proj",
            dtype=self.dtype_policy,
        )
        self.proj.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.proj(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.classes,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "classes": self.classes,
                "kernel_size": self.kernel_size,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )

        return config


@register_keras_serializable(package="SegMe>Common")
class ClassificationActivation(layers.Layer):
    def __init__(self, dtype="float32", **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        activation = "softmax" if channels > 1 else "sigmoid"
        self.act = layers.Activation(
            activation, dtype=self.dtype_policy, name="act"
        )  # for mixed_float16

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.act(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


@register_keras_serializable(package="SegMe>Common")
class ClassificationHead(layers.Layer):
    def __init__(
        self,
        classes,
        kernel_size=1,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)
        self.classes = classes
        self.kernel_size = kernel_size
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        self.proj = HeadProjection(
            self.classes,
            kernel_size=self.kernel_size,
            kernel_initializer=self.kernel_initializer,
            name="proj",
            dtype=self.dtype_policy,
        )
        self.proj.build(input_shape)

        self.act = ClassificationActivation(name="act")

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.proj(inputs)
        outputs = self.act(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.classes,)

    def compute_output_spec(self, input_spec):
        output_spec = self.proj.compute_output_spec(input_spec)
        output_spec = self.act.compute_output_spec(output_spec)

        return output_spec

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "classes": self.classes,
                "kernel_size": self.kernel_size,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )

        return config


@register_keras_serializable(package="SegMe>Common")
class ClassificationUncertainty(layers.Layer):
    def __init__(self, ord, from_logits, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=2)

        if ord not in {1, 2}:
            raise ValueError(
                f"Argument `ord` expected to be `1` or `2`. Got: {ord}"
            )

        self.ord = ord
        self.from_logits = from_logits

    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.channels})

        if self.from_logits:
            self.class_act = ClassificationActivation(dtype=self.compute_dtype)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if self.from_logits:
            inputs = self.class_act(inputs)

        if 2 == self.ord and 1 == self.channels:
            return inputs * (1.0 - inputs) * 4.0

        if 1 == self.channels:
            inputs = ops.concatenate([1.0 - inputs, inputs], axis=-1)

        scores, _ = ops.top_k(inputs, k=2)

        if 2 == self.ord:
            uncertainty = scores[..., :1] * scores[..., 1:] * 4.0
        else:
            uncertainty = 1.0 - (scores[..., :1] - scores[..., 1:])

        return uncertainty

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)

    def get_config(self):
        config = super().get_config()
        config.update({"ord": self.ord, "from_logits": self.from_logits})

        return config
