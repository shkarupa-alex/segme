from tensorflow.keras import layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from .conv2nv1 import Conv2nV1
from .conv3nv1 import Conv3nV1


@utils.register_keras_serializable(package='SegMe>MINet')
class AIM(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4) for _ in range(5)]

        if not isinstance(filters, (list, tuple)) or len(filters) != 5:
            raise ValueError('Parameter "filters" should contain 5 filter values')

        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        self.conv0 = Conv2nV1(filters=self.filters[0], main=0)
        self.conv1 = Conv3nV1(filters=self.filters[1])
        self.conv2 = Conv3nV1(filters=self.filters[2])
        self.conv3 = Conv3nV1(filters=self.filters[3])
        self.conv4 = Conv2nV1(filters=self.filters[4], main=1)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        out0 = self.conv0([inputs[0], inputs[1]])
        out1 = self.conv1([inputs[0], inputs[1], inputs[2]])
        out2 = self.conv2([inputs[1], inputs[2], inputs[3]])
        out3 = self.conv3([inputs[2], inputs[3], inputs[4]])
        out4 = self.conv4([inputs[3], inputs[4]])

        return out0, out1, out2, out3, out4

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        out_shape0 = self.conv0.compute_output_shape([input_shape[0], input_shape[1]])
        out_shape1 = self.conv1.compute_output_shape([input_shape[0], input_shape[1], input_shape[2]])
        out_shape2 = self.conv2.compute_output_shape([input_shape[1], input_shape[2], input_shape[3]])
        out_shape3 = self.conv3.compute_output_shape([input_shape[2], input_shape[3], input_shape[4]])
        out_shape4 = self.conv4.compute_output_shape([input_shape[3], input_shape[4]])

        return out_shape0, out_shape1, out_shape2, out_shape3, out_shape4

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config
