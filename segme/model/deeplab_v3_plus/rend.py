import tensorflow as tf
from tensorflow.keras import Model, layers, losses, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from .model import DeepLabV3Plus
from ...common import HeadProjection, PointRend, PointLoss


@utils.register_keras_serializable(package='SegMe>DeepLabV3Plus')
class DeepLabV3PlusWithPointRend(DeepLabV3Plus):
    def __init__(
            self, classes, bone_arch, bone_init, bone_train, aspp_filters, aspp_stride, low_filters, decoder_filters,
            rend_strides, rend_units, rend_points, rend_oversample, rend_importance, **kwargs):
        super().__init__(
            classes=classes, bone_arch=bone_arch, bone_init=bone_init, bone_train=bone_train, aspp_filters=aspp_filters,
            aspp_stride=aspp_stride, low_filters=low_filters, decoder_filters=decoder_filters, **kwargs)

        self.rend_strides = rend_strides
        self.rend_units = rend_units
        self.rend_points = rend_points
        self.rend_oversample = rend_oversample
        self.rend_importance = rend_importance

        self.add_strides = rend_strides

    @shape_type_conversion
    def build(self, input_shape):
        self.rend = PointRend(
            self.classes, self.rend_units, self.rend_points, self.rend_oversample, self.rend_importance,
            fines=len(self.rend_strides), residual=False, align_corners=True)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs, _, *rend_feats = self._call(inputs)
        outputs = self.rend([inputs, outputs, *rend_feats])

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.rend.compute_output_shape([input_shape])

    def compute_output_signature(self, input_signature):
        return self.rend.compute_output_signature([input_signature])

    def get_config(self):
        config = super().get_config()
        config.update({
            'rend_strides': self.rend_strides,
            'rend_units': self.rend_units,
            'rend_points': self.rend_points,
            'rend_oversample': self.rend_oversample,
            'rend_importance': self.rend_importance
        })

        return config


def build_deeplab_v3_plus_with_point_rend(
        classes, bone_arch, bone_init, bone_train, rend_weights, aspp_filters=256, aspp_stride=16,
        low_filters=48, decoder_filters=256, rend_strides=(2, 4), rend_units=(256, 256, 256), rend_points=(0.008, 0.06),
        rend_oversample=3, rend_importance=0.75, rend_reduction=losses.Reduction.AUTO):
    model_inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')

    rend_inputs = [layers.Input(name='label', shape=[None, None, 1], dtype='int32')]
    tf.get_logger().warning('Don\'t forget to pass "label" input into features')

    if rend_weights:
        rend_inputs.append(layers.Input(name='weight', shape=[None, None, 1], dtype='float32'))
        tf.get_logger().warning('Don\'t forget to pass "label" input into features')

    outputs, point_logits, point_coords = DeepLabV3PlusWithPointRend(
        classes, bone_arch=bone_arch, bone_init=bone_init, bone_train=bone_train, aspp_filters=aspp_filters,
        aspp_stride=aspp_stride, low_filters=low_filters, decoder_filters=decoder_filters, rend_strides=rend_strides,
        rend_units=rend_units, rend_points=rend_points, rend_oversample=rend_oversample,
        rend_importance=rend_importance)(model_inputs)
    model = Model(inputs=[model_inputs, *rend_inputs], outputs=outputs, name='deeplab_v3_plus_with_point_rend')

    point_loss = PointLoss(classes, weighted=rend_weights, reduction=rend_reduction)(
        [point_logits, point_coords, *rend_inputs])
    model.add_loss(point_loss)

    return model
