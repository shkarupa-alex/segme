import tensorflow as tf
from keras import layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tensorflow_addons import layers as addon_layers
from ...common import ConvNormRelu, SameConv, ResizeByScale


@register_keras_serializable(package='SegMe>MatteFormer')
class Decoder(layers.Layer):
    def __init__(self, filters, depths, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4) for _ in range(6)]

        self.filters = filters
        self.depths = depths

    @shape_type_conversion
    def build(self, input_shape):
        self.refine8 = models.Sequential([
            ConvNormRelu(32, 3, activation='leaky_relu'),
            SameConv(1, 3),
            ResizeByScale(8.),
            layers.Activation('tanh', dtype='float32')
        ])
        self.refine4 = models.Sequential([
            ConvNormRelu(32, 3, activation='leaky_relu'),
            SameConv(1, 3),
            ResizeByScale(4.),
            layers.Activation('tanh', dtype='float32')
        ])
        self.refine1 = models.Sequential([
            ConvNormRelu(32, 3, activation='leaky_relu'),
            SameConv(1, 3),
            layers.Activation('tanh', dtype='float32')
        ])

        self.layer1 = models.Sequential([
            Block(self.filters[0], 1 if i > 0 else 2) for i in range(self.depths[0])])
        self.layer2 = models.Sequential([
            Block(self.filters[1], 1 if i > 0 else 2) for i in range(self.depths[1])])
        self.layer3 = models.Sequential([
            Block(self.filters[2], 1 if i > 0 else 2) for i in range(self.depths[2])])
        self.layer4 = models.Sequential([
            Block(self.filters[3], 1 if i > 0 else 2) for i in range(self.depths[3])])
        self.layer5 = models.Sequential([
            addon_layers.SpectralNormalization(
                layers.Conv2DTranspose(32, 4, strides=2, padding='same', use_bias=False)),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
        ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        feats1, feats2, feats4, feats8, feats16, feats32 = inputs

        outputs = self.layer1(feats32) + feats16
        outputs = self.layer2(outputs) + feats8
        outputs8 = self.refine8(outputs)

        outputs = self.layer3(outputs) + feats4
        outputs4 = self.refine4(outputs)

        outputs = self.layer4(outputs) + feats2
        outputs = self.layer5(outputs) + feats1
        outputs1 = self.refine1(outputs)

        outputs1 = outputs1 / 2. + .5
        outputs4 = outputs4 / 2. + .5
        outputs8 = outputs8 / 2. + .5

        return outputs1, outputs4, outputs8

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return (input_shape[0][:-1] + (1,), ) * 3

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)

        return [tf.TensorSpec(dtype='float32', shape=os.shape) for os in outptut_signature]

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'depths': self.depths
        })

        return config


@register_keras_serializable(package='SegMe>MatteFormer')
class Block(layers.Layer):
    def __init__(self, filters, stride, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        if stride not in {1, 2}:
            raise ValueError('Unsupported stride')

        self.filters = filters
        self.stride = stride

    @shape_type_conversion
    def build(self, input_shape):
        if 1 == self.stride:
            conv1 = SameConv(self.filters, 3)
        else:
            conv1 = layers.Conv2DTranspose(self.filters, 4, strides=self.stride, padding='same', use_bias=False)
        self.conv1 = addon_layers.SpectralNormalization(conv1)
        self.bn1 = layers.BatchNormalization()
        self.act = layers.LeakyReLU(0.2)
        self.conv2 = addon_layers.SpectralNormalization(SameConv(self.filters, 3))
        self.bn2 = layers.BatchNormalization()

        self.upsample = None
        if 2 == self.stride:
            self.upsample = models.Sequential([
                ResizeByScale(2.),
                addon_layers.SpectralNormalization(layers.Conv2D(self.filters, 1)),
                layers.BatchNormalization()
            ])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.act(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)

        if self.upsample is None:
            identity = inputs
        else:
            identity = self.upsample(inputs)

        outputs = self.act(outputs + identity)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = self.conv1.compute_output_shape(input_shape)
        output_shape = self.conv2.compute_output_shape(output_shape)

        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'stride': self.stride
        })

        return config
