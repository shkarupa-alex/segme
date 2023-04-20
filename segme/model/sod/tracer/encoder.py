from keras import models, layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion
from segme.policy import bbpol
from segme.policy.backbone.utils import get_layer
from segme.model.sod.tracer.edge import FrequencyEdge


@register_keras_serializable(package='SegMe>Model>SOD>Tracer')
class Encoder(layers.Layer):
    def __init__(self, radius, confidence, policy=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.radius = radius
        self.confidence = confidence
        self.policy = bbpol.deserialize(policy or bbpol.global_policy())

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: channels})

        self.edgatt = FrequencyEdge(self.radius, self.confidence)

        base_model = bbpol.BACKBONES.new(self.policy.arch_type, self.policy.init_type, channels, [4, 8, 16, 32])
        _, all_endpoints = bbpol.BACKBONES[self.policy.arch_type]
        stride4_point = all_endpoints[2]

        stage4 = models.Model(inputs=base_model.inputs, outputs=get_layer(base_model, stride4_point))
        stageX = models.Model(inputs=get_layer(base_model, stride4_point), outputs=base_model.outputs[1:])

        dtype = base_model.get_layer(index=0).dtype
        inputs = layers.Input(name='image', shape=[None, None, channels], dtype=dtype)
        feats4 = stage4(inputs)
        feats4, edges = self.edgatt(feats4)
        featsX = stageX(feats4)

        self.bone = models.Model(inputs=inputs, outputs=[edges, feats4] + featsX)
        self.bone.trainable = True

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.bone(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.bone.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'radius': self.radius,
            'confidence': self.confidence,
            'policy': bbpol.serialize(self.policy)
        })

        return config
