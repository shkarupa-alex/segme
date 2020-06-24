from tensorflow.keras import layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from ...common import ASPP
from ...backbone import Backbone


@utils.register_keras_serializable(package='SegMe')
class DeepLabV3PlusEncoder(layers.Layer):
    def __init__(self, bone_arch, bone_init, bone_train, aspp_filters,
                 aspp_stride, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')
        self.bone_arch = bone_arch
        self.bone_init = bone_init
        self.bone_train = bone_train
        self.aspp_filters = aspp_filters
        self.aspp_stride = aspp_stride

    @shape_type_conversion
    def build(self, input_shape):
        self.bone = Backbone(
            self.bone_arch, self.bone_init, self.bone_train,
            scales=[4, self.aspp_stride])
        self.aspp = ASPP(self.aspp_filters, self.aspp_stride)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        low_feats, high_feats = self.bone(inputs)
        aspp_feats = self.aspp(high_feats)

        return low_feats, aspp_feats

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        low_feats_shape, high_feats_shape = \
            self.bone.compute_output_shape(input_shape)
        high_feats_shape = high_feats_shape[:-1] + (self.aspp_filters,)

        return low_feats_shape, high_feats_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'bone_arch': self.bone_arch,
            'bone_init': self.bone_init,
            'bone_train': self.bone_train,
            'aspp_filters': self.aspp_filters,
            'aspp_stride': self.aspp_stride
        })

        return config
