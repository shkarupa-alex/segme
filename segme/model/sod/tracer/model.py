import tensorflow as tf
from tf_keras import layers, models
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.tf_utils import shape_type_conversion
from segme.common.head import ClassificationActivation
from segme.common.resize import BilinearInterpolation
from segme.model.sod.tracer.aggr import Aggregation
from segme.model.sod.tracer.encoder import Encoder
from segme.model.sod.tracer.objatt import ObjectAttention
from segme.model.sod.tracer.recfield import ReceptiveField


@register_keras_serializable(package='SegMe>Model>SOD>Tracer')
class Tracer(layers.Layer):
    def __init__(self, radius, confidence, rfb, denoise, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')

        self.radius = radius
        self.confidence = confidence
        self.rfb = rfb
        self.denoise = denoise

    @shape_type_conversion
    def build(self, input_shape):
        self.encoder = Encoder(self.radius, self.confidence)

        self.rfb8 = ReceptiveField(self.rfb[0])
        self.rfb16 = ReceptiveField(self.rfb[1])
        self.rfb32 = ReceptiveField(self.rfb[2])

        self.agg = Aggregation(self.confidence)

        self.up2 = BilinearInterpolation(2)
        self.up4 = BilinearInterpolation(4)
        self.up8 = BilinearInterpolation(8)

        self.objatt0 = ObjectAttention(self.denoise)
        self.objatt1 = ObjectAttention(self.denoise)

        self.act = ClassificationActivation()

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        edges, feats4, feats8, feats16, feats32 = self.encoder(inputs)
        edges = self.up4(edges)

        aggrs8 = self.agg([self.rfb8(feats8), self.rfb16(feats16), self.rfb32(feats32)])
        logits8_ag = self.up8(aggrs8)

        objatts8 = self.objatt1([feats8, aggrs8])
        logits8_oa = self.up8(objatts8)

        objatts4 = self.objatt0([feats4, self.up2(objatts8)])
        logits4_oa = self.up4(objatts4)

        logits = (logits4_oa + logits8_oa + logits8_ag) / 3.

        return self.act(logits), self.act(edges), self.act(logits8_ag), self.act(logits8_oa), self.act(logits4_oa)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:-1] + (1,)

        return (output_shape,) * 5

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)

        return tuple(tf.TensorSpec(dtype='float32', shape=os.shape) for os in outptut_signature)

    def get_config(self):
        config = super().get_config()
        config.update({
            'radius': self.radius,
            'confidence': self.confidence,
            'rfb': self.rfb,
            'denoise': self.denoise
        })

        return config


def build_tracer(radius=16, confidence=0.1, rfb=(32, 64, 128), denoise=0.93):
    inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    outputs = Tracer(radius=radius, confidence=confidence, rfb=rfb, denoise=denoise)(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='tracer')

    return model
