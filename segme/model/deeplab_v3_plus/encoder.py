from tensorflow.keras import layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from .aspp import ASPP
from ...backbone import Backbone


@utils.register_keras_serializable(package='SegMe>DeepLabV3Plus')
class Encoder(layers.Layer):
    def __init__(self, bone_arch, bone_init, bone_train, aspp_filters, aspp_stride, ret_strides=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')
        self.bone_arch = bone_arch
        self.bone_init = bone_init
        self.bone_train = bone_train
        self.aspp_filters = aspp_filters
        self.aspp_stride = aspp_stride
        self.low_stride = 4
        self.ret_strides = ret_strides

    @shape_type_conversion
    def build(self, input_shape):
        self.scales = [self.low_stride, self.aspp_stride]
        if self.ret_strides is not None:
            self.scales.extend(self.ret_strides)
            self.scales = sorted(set(self.scales))

        self.bone = Backbone(self.bone_arch, self.bone_init, self.bone_train, scales=self.scales)
        self.aspp = ASPP(self.aspp_filters, self.aspp_stride)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        all_feats = self.bone(inputs)
        low_feats = all_feats[self.scales.index(self.low_stride)]
        high_feats = all_feats[self.scales.index(self.aspp_stride)]
        aspp_feats = self.aspp(high_feats)

        if self.ret_strides is None:
            return low_feats, aspp_feats

        ret_feats = [all_feats[self.scales.index(stride)] for stride in self.ret_strides]

        return (low_feats, aspp_feats, *ret_feats)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        all_feats_shape = self.bone.compute_output_shape(input_shape)
        low_feats_shape = all_feats_shape[self.scales.index(self.low_stride)]
        high_feats_shape = all_feats_shape[self.scales.index(self.aspp_stride)]
        aspp_feats_shape = high_feats_shape[:-1] + (self.aspp_filters,)

        if self.ret_strides is None:
            return low_feats_shape, aspp_feats_shape

        ret_feats_shape = [all_feats_shape[self.scales.index(stride)] for stride in self.ret_strides]

        return (low_feats_shape, aspp_feats_shape, *ret_feats_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'bone_arch': self.bone_arch,
            'bone_init': self.bone_init,
            'bone_train': self.bone_train,
            'aspp_filters': self.aspp_filters,
            'aspp_stride': self.aspp_stride,
            'ret_strides': self.ret_strides
        })

        return config
