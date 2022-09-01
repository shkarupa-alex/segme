from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.policy import respol


@register_keras_serializable(package='SegMe>Common>Interpolation')
class SmoothInterpolation(layers.Layer):
    def __init__(self, scale, policy=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4) if scale is not None else [
            layers.InputSpec(ndim=4), layers.InputSpec(ndim=4)]  # targets, samples

        self.scale = None if scale is None else float(scale)
        self.policy = respol.deserialize(policy or respol.global_policy())

    @shape_type_conversion
    def build(self, input_shape):
        self.resize = respol.RESIZERS.new(self.policy.name, self.scale)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.resize(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.resize.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'scale': self.scale,
            'policy': respol.serialize(self.policy)
        })

        return config
