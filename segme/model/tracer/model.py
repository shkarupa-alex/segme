import tensorflow as tf
from keras import layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .aggr import Aggregation
from .edge import FrequencyEdge
from .objatt import ObjectAttention
from .recfield import ReceptiveField
from ...backbone import Backbone
from ...common import HeadActivation, ResizeByScale


@register_keras_serializable(package='SegMe>Tracer')
class Tracer(layers.Layer):
    def __init__(self, bone_arch, bone_init, bone_train, radius, confidence, rfb, denoise, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')

        self.bone_arch = bone_arch
        self.bone_init = bone_init
        self.bone_train = bone_train

        self.radius = radius
        self.confidence = confidence
        self.rfb = rfb
        self.denoise = denoise

    @shape_type_conversion
    def build(self, input_shape):
        self.bone = Backbone(self.bone_arch, self.bone_init, self.bone_train, scales=[4, 8, 16, 32])

        self.edgatt = FrequencyEdge(self.radius, self.confidence)

        self.rfb8 = ReceptiveField(self.rfb[0])
        self.rfb16 = ReceptiveField(self.rfb[1])
        self.rfb32 = ReceptiveField(self.rfb[2])

        self.agg = Aggregation(self.confidence)

        self.up2 = ResizeByScale(2)
        self.up4 = ResizeByScale(4)
        self.up8 = ResizeByScale(8)

        self.objatt0 = ObjectAttention(3, self.denoise)
        self.objatt1 = ObjectAttention(3, self.denoise)

        self.act = HeadActivation(1)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        feats4, feats8, feats16, feats32 = self.bone(inputs)

        feats4, edges = self.edgatt(feats4)
        edges = self.up4(edges)

        aggrs8 = self.agg([self.rfb8(feats8), self.rfb16(feats16), self.rfb32(feats32)])

        logits8 = self.up8(aggrs8)

        objatts8 = self.objatt1([feats8, aggrs8])
        logits8_oa = self.up8(objatts8)

        objatts4 = self.objatt0([feats4, self.up2(objatts8)])
        logits4_oa = self.up4(objatts4)

        logits = (logits4_oa + logits8_oa + logits8) / 3

        return self.act(logits), self.act(edges), self.act(logits8), self.act(logits8_oa), self.act(logits4_oa)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:-1] + (1,)

        return [output_shape] * 5

    def compute_output_signature(self, input_signature):
        output_shape = input_signature.shape[:-1] + (1,)
        output_signature = tf.TensorSpec(dtype='float32', shape=output_shape)

        return [output_signature] * 5

    def get_config(self):
        config = super().get_config()
        config.update({
            'bone_arch': self.bone_arch,
            'bone_init': self.bone_init,
            'bone_train': self.bone_train,
            'radius': self.radius,
            'confidence': self.confidence,
            'rfb': self.rfb,
            'denoise': self.denoise
        })

        return config


def build_tracer(
        bone_arch='resnet_50', bone_init='imagenet', bone_train=False, radius=16, confidence=0.1,
        rfb=(32, 64, 128), denoise=0.93):
    inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    outputs = Tracer(bone_arch=bone_arch, bone_init=bone_init, bone_train=bone_train, radius=radius,
                     confidence=confidence, rfb=rfb, denoise=denoise)(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='tracer')

    return model
