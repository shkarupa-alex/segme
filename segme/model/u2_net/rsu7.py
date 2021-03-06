from tensorflow.keras import layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from ...common import ConvBnRelu, resize_by_sample


@utils.register_keras_serializable(package='SegMe>U2Net')
class RSU7(layers.Layer):
    def __init__(self, mid_features=12, out_features=3, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.mid_features = mid_features
        self.out_features = out_features

    @shape_type_conversion
    def build(self, input_shape):
        self.cbr0 = ConvBnRelu(self.out_features, 3)

        self.cbr1 = ConvBnRelu(self.mid_features, 3)
        self.pool1 = layers.MaxPool2D(2, padding='same')

        self.cbr2 = ConvBnRelu(self.mid_features, 3)
        self.pool2 = layers.MaxPool2D(2, padding='same')

        self.cbr3 = ConvBnRelu(self.mid_features, 3)
        self.pool3 = layers.MaxPool2D(2, padding='same')

        self.cbr4 = ConvBnRelu(self.mid_features, 3)
        self.pool4 = layers.MaxPool2D(2, padding='same')

        self.cbr5 = ConvBnRelu(self.mid_features, 3)
        self.pool5 = layers.MaxPool2D(2, padding='same')

        self.cbr6 = ConvBnRelu(self.mid_features, 3)

        self.cbr7 = ConvBnRelu(self.mid_features, 3, dilation_rate=2)

        self.cbr6d = ConvBnRelu(self.mid_features, 3)
        self.cbr5d = ConvBnRelu(self.mid_features, 3)
        self.cbr4d = ConvBnRelu(self.mid_features, 3)
        self.cbr3d = ConvBnRelu(self.mid_features, 3)
        self.cbr2d = ConvBnRelu(self.mid_features, 3)
        self.cbr1d = ConvBnRelu(self.out_features, 3)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = inputs
        outputs0 = self.cbr0(outputs)

        outputs1 = self.cbr1(outputs0)
        outputs = self.pool1(outputs1)

        outputs2 = self.cbr2(outputs)
        outputs = self.pool2(outputs2)

        outputs3 = self.cbr3(outputs)
        outputs = self.pool3(outputs3)

        outputs4 = self.cbr4(outputs)
        outputs = self.pool4(outputs4)

        outputs5 = self.cbr5(outputs)
        outputs = self.pool5(outputs5)

        outputs6 = self.cbr6(outputs)

        outputs7 = self.cbr7(outputs6)

        outputs6d = self.cbr6d(layers.concatenate([outputs7, outputs6]))
        outputs6dup = resize_by_sample([outputs6d, outputs5])

        outputs5d = self.cbr5d(layers.concatenate([outputs6dup, outputs5]))
        outputs5dup = resize_by_sample([outputs5d, outputs4])

        outputs4d = self.cbr4d(layers.concatenate([outputs5dup, outputs4]))
        outputs4dup = resize_by_sample([outputs4d, outputs3])

        outputs3d = self.cbr3d(layers.concatenate([outputs4dup, outputs3]))
        outputs3dup = resize_by_sample([outputs3d, outputs2])

        outputs2d = self.cbr2d(layers.concatenate([outputs3dup, outputs2]))
        outputs2dup = resize_by_sample([outputs2d, outputs1])

        outputs1d = self.cbr1d(layers.concatenate([outputs2dup, outputs1]))

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
