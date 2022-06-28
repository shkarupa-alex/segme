import itertools
import tensorflow as tf
from keras import backend, layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .util import make_coord
from ...common import ConvNormRelu, GridSample, resize_by_sample


@register_keras_serializable(package='SegMe>HqsCrm')
class Decoder(layers.Layer):
    def __init__(self, aspp_filters, aspp_drop, mlp_units, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4) for _ in range(3)] + [
            layers.InputSpec(ndim=4, axes={-1: 2}) for _ in range(2)]

        self.aspp_filters = aspp_filters
        self.aspp_drop = aspp_drop
        self.mlp_units = mlp_units

    @shape_type_conversion
    def build(self, input_shape):
        self.aspp2 = ConvNormRelu(self.aspp_filters[0], 1)  # TODO: try ASPP
        self.aspp4 = ConvNormRelu(self.aspp_filters[1], 1)
        self.aspp32 = ConvNormRelu(self.aspp_filters[2], 1)
        self.fuse = ConvNormRelu(sum(self.aspp_filters), 3)
        self.drop = layers.Dropout(self.aspp_drop)

        self.sample = GridSample(mode='nearest', align_corners=False)
        self.mlp = models.Sequential(
            [layers.Dense(u, activation='relu') for u in self.mlp_units] + [layers.Dense(1)])

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        feats2, feats4, feats32, coords, cells = inputs

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

        epsilon = tf.cast(backend.epsilon(), self.compute_dtype)

        batch, height, width, _ = tf.unstack(tf.shape(aspp))
        size = tf.cast([height, width], self.compute_dtype)
        rxry = 1. / size

        feat_coords = make_coord(batch, height, width, dtype=self.compute_dtype)
        rel_cells = cells * size

        preds, areas = [], []
        for vxvy in itertools.product([-1., 1.], [-1., 1.]):
            coords_ = coords + vxvy * rxry + epsilon
            coords_ = tf.clip_by_value(coords_, -1 + epsilon, -epsilon + 1)
            coords_ = tf.reverse(coords_, axis=[-1])

            feat_samp = self.sample([aspp, coords_])
            coord_samp = self.sample([feat_coords, coords_])
            rel_coords = (coords - coord_samp) * size

            pred = tf.concat([feat_samp, rel_coords, coords, rel_cells], axis=-1)
            pred = self.mlp(pred)
            preds.append(pred)

            area = tf.reduce_prod(rel_coords, axis=-1, keepdims=True)
            areas.insert(0, tf.abs(area) + epsilon)

        outputs = [pred * area for pred, area in zip(preds, areas)]
        outputs = tf.math.add_n(outputs) / tf.math.add_n(areas)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[3][:-1] + (1,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'mlp_units': self.mlp_units,
            'aspp_drop': self.aspp_drop,
            'aspp_filters': self.aspp_filters
        })

        return config
