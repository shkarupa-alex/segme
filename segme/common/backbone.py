from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.policy import bbpol


@register_keras_serializable(package='SegMe>Common')
class Backbone(layers.Layer):
    def __init__(self, scales=None, policy=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.scales = scales
        self.policy = bbpol.deserialize(policy or bbpol.global_policy())

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: channels})

        self.bone = bbpol.BACKBONES.new(self.policy.arch_type, self.policy.init_type, channels, self.scales)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.bone(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.bone.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'scales': self.scales,
            'policy': bbpol.serialize(self.policy)
        })

        return config
