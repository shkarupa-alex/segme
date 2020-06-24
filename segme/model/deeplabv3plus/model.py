from tensorflow.keras import layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from .encoder import DeepLabV3PlusEncoder
from .decoder import DeepLabV3PlusDecoder


@utils.register_keras_serializable(package='SegMe')
class DeepLabV3Plus(layers.Layer):
    """ Reference: https://arxiv.org/pdf/1802.02611.pdf """
    def __init__(self,
                 classes,
                 bone_arch,
                 bone_init,
                 bone_freeze,
                 aspp_filters=256,
                 aspp_stride=16,
                 low_filters=48,
                 decoder_filters=256,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')
        self.classes = classes
        self._classes = self.classes if self.classes > 2 else 1
        self.bone_arch = bone_arch
        self.bone_init = bone_init
        self.bone_freeze = bone_freeze
        self.aspp_filters = aspp_filters
        self.aspp_stride = aspp_stride
        self.low_filters = low_filters
        self.decoder_filters = decoder_filters

    @shape_type_conversion
    def build(self, input_shape):
        self.enc = DeepLabV3PlusEncoder(
            self.bone_arch, self.bone_init, self.bone_freeze,
            self.aspp_filters, self.aspp_stride)
        self.dec = DeepLabV3PlusDecoder(self.low_filters, self.decoder_filters)

        activation = 'softmax' if self.classes > 2 else 'sigmoid'
        self.pred = layers.Conv2D(
            self._classes, 1, padding='same', activation=activation)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        low_feats, high_feats = self.enc(inputs)
        outputs = self.dec([inputs, low_feats, high_feats])
        outputs = self.pred(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self._classes,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'classes': self.classes,
            'bone_arch': self.bone_arch,
            'bone_init': self.bone_init,
            'bone_freeze': self.bone_freeze,
            'aspp_filters': self.aspp_filters,
            'aspp_stride': self.aspp_stride,
            'low_filters': self.low_filters,
            'decoder_filters': self.decoder_filters
        })

        return config
