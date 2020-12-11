from tensorflow.keras import Model, layers, losses, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from .encoder import Encoder
from .decoder import Decoder
from ...common import HeadProjection, PointRend, point_sample


@utils.register_keras_serializable(package='SegMe>DeepLabV3Plus')
class DeepLabV3PlusWithPointRend(layers.Layer):
    def __init__(
            self, classes, bone_arch, bone_init, bone_train, aspp_filters, aspp_stride, low_filters, decoder_filters,
            rend_strides, rend_units, rend_points, rend_oversample, rend_importance, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')

        self.classes = classes
        self.bone_arch = bone_arch
        self.bone_init = bone_init
        self.bone_train = bone_train
        self.aspp_filters = aspp_filters
        self.aspp_stride = aspp_stride
        self.low_filters = low_filters
        self.decoder_filters = decoder_filters
        self.rend_strides = rend_strides
        self.rend_units = rend_units
        self.rend_points = rend_points
        self.rend_oversample = rend_oversample
        self.rend_importance = rend_importance

    @shape_type_conversion
    def build(self, input_shape):
        self.enc = Encoder(
            self.bone_arch, self.bone_init, self.bone_train, self.aspp_filters, self.aspp_stride,
            ret_strides=self.rend_strides)
        self.dec = Decoder(self.low_filters, self.decoder_filters)
        self.proj = HeadProjection(self.classes)

        self.rend = PointRend(
            self.classes, self.rend_units, self.rend_points, self.rend_oversample, self.rend_importance,
            fines=len(self.rend_strides), residual=False, align_corners=True)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        low_feats, high_feats, *rend_feats = self.enc(inputs)
        dec_feats = self.dec([low_feats, high_feats])
        coarse_feats = self.proj(dec_feats)

        return self.rend([inputs, coarse_feats, *rend_feats])

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.rend.compute_output_shape([input_shape])

    def compute_output_signature(self, input_signature):
        return self.rend.compute_output_signature([input_signature])

    def get_config(self):
        config = super().get_config()
        config.update({
            'classes': self.classes,
            'bone_arch': self.bone_arch,
            'bone_init': self.bone_init,
            'bone_train': self.bone_train,
            'aspp_filters': self.aspp_filters,
            'aspp_stride': self.aspp_stride,
            'low_filters': self.low_filters,
            'decoder_filters': self.decoder_filters,
            'rend_strides': self.rend_strides,
            'rend_units': self.rend_units,
            'rend_points': self.rend_points,
            'rend_oversample': self.rend_oversample,
            'rend_importance': self.rend_importance
        })

        return config


def build_deeplab_v3_plus_with_point_rend(
        channels, classes, bone_arch, bone_init, bone_train, aspp_filters=256, aspp_stride=16, low_filters=48,
        decoder_filters=256, rend_strides=(2, 4), rend_units=(256, 256, 256), rend_points=(1024, 8192),
        rend_oversample=3, rend_importance=0.75, rend_weights=False, rend_reduction=losses.Reduction.AUTO):
    norm_inputs = layers.Input(name='image', shape=[None, None, channels], dtype='uint8')
    rend_inputs = [layers.Input(name='label', shape=[None, None, 1], dtype='int32')]
    if rend_weights:
        rend_inputs.append(layers.Input(name='weight', shape=[None, None, 1], dtype='float32'))

    outputs, point_logits, point_coords = DeepLabV3PlusWithPointRend(
        classes, bone_arch=bone_arch, bone_init=bone_init, bone_train=bone_train, aspp_filters=aspp_filters,
        aspp_stride=aspp_stride, low_filters=low_filters, decoder_filters=decoder_filters, rend_strides=rend_strides,
        rend_units=rend_units, rend_points=rend_points, rend_oversample=rend_oversample,
        rend_importance=rend_importance)(norm_inputs)
    model = Model(inputs=[norm_inputs] + rend_inputs, outputs=outputs, name='deeplab_v3_plus_with_point_rend')

    point_labels = point_sample([rend_inputs[0], point_coords], align_corners=True, mode='nearest')
    point_weights = None if not rend_weights else point_sample(
        [rend_inputs[1], point_coords], align_corners=True, mode='nearest')

    loss_class = losses.BinaryCrossentropy if 1 == classes else losses.SparseCategoricalCrossentropy
    point_loss = loss_class(reduction=rend_reduction, from_logits=True)
    model.add_loss(point_loss(point_labels, point_logits, sample_weight=point_weights))

    return model
