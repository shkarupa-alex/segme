from keras import layers, models
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.backbone import Backbone
from segme.common.convnormact import ConvNormAct
from segme.common.head import ClassificationHead
from segme.common.interrough import BilinearInterpolation
from segme.model.sod.minet.aim import AIM
from segme.model.sod.minet.sim import SIM


@register_keras_serializable(package='SegMe>Model>SOD>MINet')
class MINet(layers.Layer):
    def __init__(self, classes, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')
        self.classes = classes

    @shape_type_conversion
    def build(self, input_shape):
        self.bone = Backbone([2, 4, 8, 16, 32])

        self.trans = AIM(filters=(64, 64, 64, 64, 64))

        self.sim32 = SIM(32)
        self.sim16 = SIM(32)
        self.sim8 = SIM(32)
        self.sim4 = SIM(32)
        self.sim2 = SIM(32)

        self.upconv32 = ConvNormAct(64, 3)
        self.upconv16 = ConvNormAct(64, 3)
        self.upconv8 = ConvNormAct(64, 3)
        self.upconv4 = ConvNormAct(64, 3)
        self.upconv2 = ConvNormAct(32, 3)
        self.upconv1 = ConvNormAct(32, 3)

        self.resize = BilinearInterpolation(None)

        self.head = ClassificationHead(self.classes)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        c1, c2, c3, c4, c5 = self.bone(inputs)

        out1, out2, out3, out4, out5 = self.trans([c1, c2, c3, c4, c5])

        out5 = self.upconv32(self.sim32(out5) + out5)

        out4 = self.resize([out5, out4]) + out4
        out4 = self.upconv16(self.sim16(out4) + out4)

        out3 = self.resize([out4, out3]) + out3
        out3 = self.upconv8(self.sim8(out3) + out3)

        out2 = self.resize([out3, out2]) + out2
        out2 = self.upconv4(self.sim4(out2) + out2)

        out1 = self.resize([out2, out1]) + out1
        out1 = self.upconv2(self.sim2(out1) + out1)

        outputs = self.upconv1(self.resize([out1, inputs]))

        return self.head(outputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.head.compute_output_shape(input_shape)

    def compute_output_signature(self, input_signature):
        return self.head.compute_output_signature(input_signature)

    def get_config(self):
        config = super().get_config()
        config.update({'classes': self.classes})

        return config


def build_minet(classes):
    inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    outputs = MINet(classes)(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='minet')

    return model
