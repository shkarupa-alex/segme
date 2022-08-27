import itertools
import tensorflow as tf
from keras import backend, layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ...common import ConvNormRelu, query_features, resize_by_sample


@register_keras_serializable(package='SegMe>HqsCrm')
class Decoder(layers.Layer):
    def __init__(self, aspp_filters, aspp_drop, mlp_units, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4) for _ in range(3)] + [layers.InputSpec(ndim=4, axes={-1: 2})]

        self.aspp_filters = aspp_filters
        self.aspp_drop = aspp_drop
        self.mlp_units = mlp_units

    @shape_type_conversion
    def build(self, input_shape):
        self.aspp2 = ConvNormRelu(self.aspp_filters[0], 1)  # TODO: try ASPP?
        self.aspp4 = ConvNormRelu(self.aspp_filters[1], 1)
        self.aspp32 = ConvNormRelu(self.aspp_filters[2], 1)
        self.fuse = ConvNormRelu(sum(self.aspp_filters), 3)
        self.drop = layers.Dropout(self.aspp_drop)

        self.imnet = models.Sequential(
            [layers.Dense(u, activation='relu') for u in self.mlp_units] + [layers.Dense(1)])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        feats2, feats4, feats32, coords = inputs

        aspp2 = self.aspp2(feats2)
        aspp2 = self.drop(aspp2)

        aspp4 = self.aspp4(feats4)
        aspp4 = self.drop(aspp4)
        aspp4 = resize_by_sample([aspp4, aspp2])

        aspp32 = self.aspp32(feats32)
        aspp32 = self.drop(aspp32)
        aspp32 = resize_by_sample([aspp32, aspp2])

        aspp = tf.concat([aspp2, aspp4, aspp32], axis=-1)
        aspp = self.fuse(aspp)
        aspp = self.drop(aspp)

        # Original implementation uses cells, but this is meaningless due to input/output scale is constant
        outputs = query_features(
            aspp, coords, self.imnet, posnet=None, cells=None, feat_unfold=False, local_ensemble=True,
            dtype=self.compute_dtype)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[-1][:-1] + (1,)

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)

        return tf.TensorSpec(dtype='float32', shape=outptut_signature.shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'mlp_units': self.mlp_units,
            'aspp_drop': self.aspp_drop,
            'aspp_filters': self.aspp_filters
        })

        return config
