from keras import layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .aim import AIM
from .sim import SIM
from ...backbone import Backbone
from ...common import ConvBnRelu, ClassificationHead, resize_by_sample


@register_keras_serializable(package='SegMe>MINet')
class MINet(layers.Layer):
    def __init__(self, classes, bone_arch, bone_init, bone_train, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')
        self.classes = classes
        self.bone_arch = bone_arch
        self.bone_init = bone_init
        self.bone_train = bone_train

    @shape_type_conversion
    def build(self, input_shape):
        self.bone = Backbone(self.bone_arch, self.bone_init, self.bone_train, scales=[2, 4, 8, 16, 32])

        self.trans = AIM(filters=(64, 64, 64, 64, 64))

        self.sim32 = SIM(32)
        self.sim16 = SIM(32)
        self.sim8 = SIM(32)
        self.sim4 = SIM(32)
        self.sim2 = SIM(32)

        self.upconv32 = ConvBnRelu(64, 3)
        self.upconv16 = ConvBnRelu(64, 3)
        self.upconv8 = ConvBnRelu(64, 3)
        self.upconv4 = ConvBnRelu(64, 3)
        self.upconv2 = ConvBnRelu(32, 3)
        self.upconv1 = ConvBnRelu(32, 3)

        self.head = ClassificationHead(self.classes)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        c1, c2, c3, c4, c5 = self.bone(inputs)

        out1, out2, out3, out4, out5 = self.trans([c1, c2, c3, c4, c5])

        out5 = self.upconv32(layers.add([self.sim32(out5), out5]))

        out4 = layers.add([resize_by_sample([out5, out4]), out4])
        out4 = self.upconv16(layers.add([self.sim16(out4), out4]))

        out3 = layers.add([resize_by_sample([out4, out3]), out3])
        out3 = self.upconv8(layers.add([self.sim8(out3), out3]))

        out2 = layers.add([resize_by_sample([out3, out2]), out2])
        out2 = self.upconv4(layers.add([self.sim4(out2), out2]))

        out1 = layers.add([resize_by_sample([out2, out1]), out1])
        out1 = self.upconv2(layers.add([self.sim2(out1), out1]))

        outputs = self.upconv1(resize_by_sample([out1, inputs]))

        return self.head(outputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.head.compute_output_shape(input_shape)

    def compute_output_signature(self, input_signature):
        return self.head.compute_output_signature(input_signature)

    def get_config(self):
        config = super().get_config()
        config.update({
            'classes': self.classes,
            'bone_arch': self.bone_arch,
            'bone_init': self.bone_init,
            'bone_train': self.bone_train
        })

        return config


def build_minet(classes, bone_arch='resnet_50', bone_init='imagenet', bone_train=False):
    inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    outputs = MINet(classes, bone_arch=bone_arch, bone_init=bone_init, bone_train=bone_train)(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='minet')

    return model
