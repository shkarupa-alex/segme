from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.backbone import Backbone
from segme.model.deeplab_v3_plus.decoder import Decoder
from segme.common.head import HeadProjection, ClassificationActivation
from segme.common.interrough import BilinearInterpolation


@register_keras_serializable(package='SegMe>Model>DeepLabV3Plus')
class DeepLabV3PlusBase(layers.Layer):
    """ Reference: https://arxiv.org/pdf/1802.02611.pdf """

    def __init__(self, classes, aspp_filters, aspp_stride, low_filters, decoder_filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')

        self.classes = classes
        self.aspp_filters = aspp_filters
        self.aspp_stride = aspp_stride
        self.low_filters = low_filters
        self.decoder_filters = decoder_filters

        self._return_stride2 = False
        self._return_decfeat = False

    @shape_type_conversion
    def build(self, input_shape):
        scale2 = [2] if self._return_stride2 else []
        self.bone = Backbone(scale2 + [4, 32])
        self.dec = Decoder(self.aspp_filters, self.aspp_stride, self.low_filters, self.decoder_filters)
        self.proj = HeadProjection(self.classes)

        self.resize = BilinearInterpolation(None, dtype='float32')
        self.act = ClassificationActivation()

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        all_feats = self.bone(inputs)

        dec_feats = self.dec(all_feats[-2:])
        logits = self.proj(dec_feats)

        if not (self._return_stride2 or self._return_decfeat):
            return logits

        outputs = (logits,)
        if self._return_stride2:
            outputs += (all_feats[0],)
        if self._return_decfeat:
            outputs += (dec_feats,)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        all_feats_shape = self.bone.compute_output_shape(input_shape)
        dec_feats_shape = self.dec.compute_output_shape(all_feats_shape[-2:])
        logits_shape = self.proj.compute_output_shape(dec_feats_shape)

        if not (self._return_stride2 or self._return_decfeat):
            return logits_shape

        outputs_shape = (logits_shape,)
        if self._return_stride2:
            outputs_shape += (all_feats_shape[0],)
        if self._return_decfeat:
            outputs_shape += (dec_feats_shape,)

        return outputs_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'classes': self.classes,
            'aspp_filters': self.aspp_filters,
            'aspp_stride': self.aspp_stride,
            'low_filters': self.low_filters,
            'decoder_filters': self.decoder_filters
        })

        return config
