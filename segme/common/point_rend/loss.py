import tensorflow as tf
from keras import layers, losses
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from keras.utils.tf_utils import shape_type_conversion
from .sample import point_sample


@register_keras_serializable(package='SegMe>PointRend')
class PointLoss(layers.Layer):
    def __init__(self, classes, weighted=False, reduction=Reduction.AUTO, **kwargs):
        kwargs['autocast'] = False
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=3, axes={-1: classes}),  # point logits
            layers.InputSpec(ndim=3, axes={-1: 2}),  # point coords
            layers.InputSpec(ndim=4, axes={-1: 1})  # labels
        ]
        if weighted:
            self.input_spec.append(layers.InputSpec(ndim=4, axes={-1: 1}))  # weights

        self.classes = classes
        self.weighted = weighted
        self.reduction = reduction

    @shape_type_conversion
    def build(self, input_shape):
        loss_class = losses.BinaryCrossentropy if 1 == self.classes else losses.SparseCategoricalCrossentropy
        self.loss_inst = loss_class(reduction=self.reduction, from_logits=True)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if self.weighted:
            logits, coords, labels, weights = inputs
            point_weights = point_sample([weights, coords], align_corners=True, mode='nearest')
        else:
            logits, coords, labels = inputs
            point_weights = None

        point_labels = point_sample([labels, coords], align_corners=True, mode='nearest')
        loss = self.loss_inst(point_labels, tf.cast(logits, 'float32'), sample_weight=point_weights)

        return loss

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if Reduction.NONE == self.reduction:
            return input_shape[0][:-1]

        return 1

    def compute_output_signature(self, input_signature):
        output_signature = super().compute_output_signature(input_signature)

        return tf.TensorSpec(dtype='float32', shape=output_signature.shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'classes': self.classes,
            'weighted': self.weighted,
            'reduction': self.reduction
        })

        return config
