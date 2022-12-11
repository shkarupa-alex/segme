from keras import layers
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import Conv, Act


@register_keras_serializable(package='SegMe>Policy>Backbone>DIY>CoMA')
class MLP(layers.Layer):
    def __init__(self, ratio, dropout, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.ratio = ratio
        self.dropout = dropout

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.fc0 = Conv(int(channels * self.ratio), 1, name='fc0')
        self.dw = Conv(None, 3, name='dw')  # From SegFormer
        self.act = Act(name='act')
        self.fc1 = Conv(channels, 1, name='fc1')
        self.drop = layers.Dropout(self.dropout, name='drop')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = self.fc0(inputs)
        outputs = self.dw(outputs)
        outputs = self.act(outputs)
        outputs = self.drop(outputs)
        outputs = self.fc1(outputs)
        outputs = self.drop(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'ratio': self.ratio,
            'dropout': self.dropout
        })

        return config
