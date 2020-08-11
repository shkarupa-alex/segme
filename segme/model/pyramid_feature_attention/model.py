from tensorflow.keras import Model, Sequential, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from .attention import SpatialAttention, ChannelWiseAttention
from .cfe import CFE
from ...backbone import Backbone
from ...common import ClassificationHead, up_by_sample_2d


@utils.register_keras_serializable(package='SegMe')
class PyramidFeatureAttention(layers.Layer):
    """ Reference: https://arxiv.org/pdf/1903.00179v2.pdf """

    def __init__(self, bone_arch, bone_init, bone_train, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')
        self.bone_arch = bone_arch
        self.bone_init = bone_init
        self.bone_train = bone_train

    @shape_type_conversion
    def build(self, input_shape):
        self.bone = Backbone(self.bone_arch, self.bone_init, self.bone_train, scales=[1, 2, 4, 8, 16])

        self.cbr0 = Sequential([
            layers.Conv2D(64, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.cbr1 = Sequential([
            layers.Conv2D(64, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.cfe0 = CFE(32)
        self.cfe1 = CFE(32)
        self.cfe2 = CFE(32)

        self.cwatt = ChannelWiseAttention()
        self.cbr2 = Sequential([
            layers.Conv2D(64, 1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.spatt = SpatialAttention()
        self.cbr3 = Sequential([
            layers.Conv2D(64, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        c1, c2, c3, c4, c5 = self.bone(inputs)

        c1 = self.cbr0(c1)
        c2 = self.cbr1(c2)

        c3_cfe = self.cfe0(c3)
        c4_cfe = self.cfe1(c4)
        c5_cfe = self.cfe2(c5)

        c5_cfe = up_by_sample_2d([c5_cfe, c3_cfe])
        c4_cfe = up_by_sample_2d([c4_cfe, c3_cfe])
        c345 = layers.concatenate([c3_cfe, c4_cfe, c5_cfe])

        c345 = self.cwatt(c345)
        c345 = self.cbr2(c345)
        c345 = up_by_sample_2d([c345, inputs])

        sa = self.spatt(c345)

        c2 = up_by_sample_2d([c2, c1])
        c12 = layers.concatenate([c1, c2])
        c12 = self.cbr3(c12)
        c12 = layers.multiply([sa, c12])

        fea = layers.concatenate([c12, c345])

        return fea

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (128,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'bone_arch': self.bone_arch,
            'bone_init': self.bone_init,
            'bone_train': self.bone_train
        })

        return config


def build_pyramid_feature_attention(channels, classes, bone_arch='vgg_16', bone_init='imagenet', bone_train=False):
    inputs = layers.Input(name='image', shape=[None, None, channels], dtype='uint8')
    outputs = PyramidFeatureAttention(bone_arch=bone_arch, bone_init=bone_init, bone_train=bone_train)(inputs)
    outputs = ClassificationHead(classes, kernel_size=3)(outputs)
    model = Model(inputs=inputs, outputs=outputs, name='pyramid_feature_attention')

    return model
