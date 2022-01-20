import tensorflow as tf
from keras import models, layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .decoder import Decoder
from ...backbone import Backbone
from ...common import ConvNormRelu, HeadActivation, HeadProjection, resize_by_sample


@register_keras_serializable(package='SegMe>F3Net')
class F3Net(layers.Layer):
    def __init__(self, classes, bone_arch, bone_init, bone_train, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')
        self.classes = classes
        self.bone_arch = bone_arch
        self.bone_init = bone_init
        self.bone_train = bone_train
        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        self.bone = Backbone(self.bone_arch, self.bone_init, self.bone_train, scales=[4, 8, 16, 32])

        self.squeeze2 = ConvNormRelu(self.filters, 1, padding='same', kernel_initializer='he_normal')
        self.squeeze3 = ConvNormRelu(self.filters, 1, padding='same', kernel_initializer='he_normal')
        self.squeeze4 = ConvNormRelu(self.filters, 1, padding='same', kernel_initializer='he_normal')
        self.squeeze5 = ConvNormRelu(self.filters, 1, padding='same', kernel_initializer='he_normal')

        self.decoder1 = Decoder(False, self.filters)
        self.decoder2 = Decoder(True, self.filters)

        self.proj_p1 = HeadProjection(self.classes, kernel_size=3, kernel_initializer='he_normal')
        self.proj_p2 = HeadProjection(self.classes, kernel_size=3, kernel_initializer='he_normal')
        self.proj_o2 = HeadProjection(self.classes, kernel_size=3, kernel_initializer='he_normal')
        self.proj_o3 = HeadProjection(self.classes, kernel_size=3, kernel_initializer='he_normal')
        self.proj_o4 = HeadProjection(self.classes, kernel_size=3, kernel_initializer='he_normal')
        self.proj_o5 = HeadProjection(self.classes, kernel_size=3, kernel_initializer='he_normal')

        self.act = HeadActivation(self.classes)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        out2h, out3h, out4h, out5v = self.bone(inputs)

        out2h = self.squeeze2(out2h)
        out3h = self.squeeze3(out3h)
        out4h = self.squeeze4(out4h)
        out5v = self.squeeze5(out5v)

        out2h, out3h, out4h, out5v, pred1 = self.decoder1([out2h, out3h, out4h, out5v])
        out2h, out3h, out4h, out5v, pred2 = self.decoder2([out2h, out3h, out4h, out5v, pred1])

        pred1 = self.proj_p1(pred1)
        pred2 = self.proj_p2(pred2)
        out2h = self.proj_o2(out2h)
        out3h = self.proj_o3(out3h)
        out4h = self.proj_o4(out4h)
        out5v = self.proj_o5(out5v)

        outputs = [pred2, pred1, out2h, out3h, out4h, out5v]
        outputs = [resize_by_sample([out, inputs]) for out in outputs]
        outputs = [self.act(out) for out in outputs]

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:-1] + (self.classes,)

        return [output_shape] * 6

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)

        return [tf.TensorSpec(dtype='float32', shape=os.shape) for os in outptut_signature]

    def get_config(self):
        config = super().get_config()
        config.update({
            'classes': self.classes,
            'bone_arch': self.bone_arch,
            'bone_init': self.bone_init,
            'bone_train': self.bone_train,
            'filters': self.filters
        })

        return config


def build_f3_net(classes, bone_arch='resnet_50', bone_init='imagenet', bone_train=False, filters=64):
    inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    outputs = F3Net(
        classes, bone_arch=bone_arch, bone_init=bone_init, bone_train=bone_train, filters=filters)(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='f3_net')

    return model
