import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers, utils
from tensorflow.python.keras.utils.control_flow_util import smart_cond
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from .model import DeepLabV3Plus
from ...common import resize_by_sample, resize_by_scale, ConvBnRelu


@utils.register_keras_serializable(package='SegMe>DeepLabV3Plus')
class DeepLabV3PlusWithHierarchicalAttention(DeepLabV3Plus):
    """ Reference: https://arxiv.org/pdf/2005.10821.pdf """

    def __init__(
            self, classes, bone_arch, bone_init, bone_train, aspp_filters, aspp_stride, low_filters, decoder_filters,
            train_scales, eval_scales, **kwargs):
        super().__init__(
            classes=classes, bone_arch=bone_arch, bone_init=bone_init, bone_train=bone_train, aspp_filters=aspp_filters,
            aspp_stride=aspp_stride, low_filters=low_filters, decoder_filters=decoder_filters, **kwargs)

        self.train_scales = train_scales
        self.eval_scales = eval_scales
        self._train_scales = sorted({1.0} | set(self.train_scales))
        self._eval_scales = sorted({1.0} | set(self.eval_scales))

    @shape_type_conversion
    def build(self, input_shape):
        self.attn = Sequential([
            ConvBnRelu(256, 3, use_bias=False),
            ConvBnRelu(256, 3, use_bias=False),
            layers.Conv2D(2, kernel_size=1, padding='same', use_bias=False, activation='sigmoid')
        ])

        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        if training is None:
            training = tf.keras.backend.learning_phase()

        outputs = smart_cond(
            training,
            lambda: self._branch(inputs, self._train_scales),
            lambda: self._branch(inputs, self._eval_scales))
        outputs = self.act(outputs)

        return outputs

    def _branch(self, inputs, scales):
        # Unlike the original implementation, we use last features before classification projection
        # https://github.com/NVIDIA/semantic-segmentation/blob/main/network/attnscale.py#L286

        # Predict 1x scale
        predictions, features = {}, {}
        predictions[1.0], features[1.0], *_ = self._call(inputs)
        predictions[1.0] = resize_by_sample([predictions[1.0], inputs])

        # Run all other scales
        for scale in scales:
            if scale == 1.0:
                continue
            resized = resize_by_scale(inputs, scale=scale)
            preds, feats, *_ = self._call(resized)
            predictions[scale] = resize_by_sample([preds, inputs])
            features[scale] = resize_by_sample([feats, features[1.0]])

        # Generate all attention outputs
        attentions = {}
        for idx in range(len(scales) - 1):
            lo_scale = scales[idx]
            hi_scale = scales[idx + 1]
            concat_feats = layers.concatenate([features[lo_scale], features[hi_scale]])
            attention = self.attn(concat_feats)
            attentions[lo_scale] = resize_by_sample([attention, inputs])

        # Normalize attentions
        norm_attn = {}
        last_attn = None
        for idx in range(len(scales) - 1):
            lo_scale = scales[idx]
            hi_scale = scales[idx + 1]
            attn_lo = attentions[lo_scale][..., 0:1]
            attn_hi = attentions[lo_scale][..., 1:2]
            if last_attn is None:
                norm_attn[lo_scale] = attn_lo
                norm_attn[hi_scale] = attn_hi
            else:
                curr_norm = last_attn / (attn_lo + attn_hi)
                norm_attn[lo_scale] = attn_lo * curr_norm
                norm_attn[hi_scale] = attn_hi * curr_norm
            last_attn = attn_hi

        # Apply attentions
        outputs = None
        for idx, scale in enumerate(scales):
            attention = norm_attn[scale]
            if outputs is None:
                outputs = predictions[scale] * attention
            else:
                outputs += predictions[scale] * attention

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'train_scales': self.train_scales,
            'eval_scales': self.eval_scales
        })

        return config


def build_deeplab_v3_plus_with_hierarchical_attention(
        classes, bone_arch, bone_init, bone_train, aspp_filters=256, aspp_stride=16, low_filters=48,
        decoder_filters=256, train_scales=(0.5,), eval_scales=(0.25, 0.5, 2.0)):
    inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    outputs = DeepLabV3PlusWithHierarchicalAttention(
        classes, bone_arch=bone_arch, bone_init=bone_init, bone_train=bone_train, aspp_filters=aspp_filters,
        aspp_stride=aspp_stride, low_filters=low_filters, decoder_filters=decoder_filters,
        train_scales=train_scales, eval_scales=eval_scales)(inputs)
    model = Model(inputs=inputs, outputs=outputs, name='deeplab_v3_plus_with_hierarchical_attention')

    return model
