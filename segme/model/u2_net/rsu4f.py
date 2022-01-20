from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ...common import ConvNormRelu


@register_keras_serializable(package='SegMe>U2Net')
class RSU4F(layers.Layer):
    def __init__(self, mid_features=12, out_features=3, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.mid_features = mid_features
        self.out_features = out_features

    @shape_type_conversion
    def build(self, input_shape):
        self.cbr0 = ConvNormRelu(self.out_features, 3, padding='same')
        self.cbr1 = ConvNormRelu(self.mid_features, 3, padding='same')
        self.cbr2 = ConvNormRelu(self.mid_features, 3, padding='same', dilation_rate=2)
        self.cbr3 = ConvNormRelu(self.mid_features, 3, padding='same', dilation_rate=4)
        self.cbr4 = ConvNormRelu(self.mid_features, 3, padding='same', dilation_rate=8)

        self.cbr3d = ConvNormRelu(self.mid_features, 3, padding='same', dilation_rate=4)
        self.cbr2d = ConvNormRelu(self.mid_features, 3, padding='same', dilation_rate=2)
        self.cbr1d = ConvNormRelu(self.out_features, 3, padding='same')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = inputs
        outputs0 = self.cbr0(outputs)
        outputs1 = self.cbr1(outputs0)
        outputs2 = self.cbr2(outputs1)
        outputs3 = self.cbr3(outputs2)
        outputs4 = self.cbr4(outputs3)

        outputs3d = self.cbr3d(layers.concatenate([outputs4, outputs3]))
        outputs2d = self.cbr2d(layers.concatenate([outputs3d, outputs2]))
        outputs1d = self.cbr1d(layers.concatenate([outputs2d, outputs1]))

        return layers.add([outputs1d, outputs0])

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.out_features,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'mid_features': self.mid_features,
            'out_features': self.out_features
        })

        return config
