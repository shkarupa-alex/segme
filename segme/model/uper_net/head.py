import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ...common import HeadProjection, HeadActivation, resize_by_sample


@register_keras_serializable(package='SegMe>UPerNet')
class Head(layers.Layer):
    def __init__(self, classes, dropout, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=4)]
        self.classes = classes
        self.dropout = dropout

    @shape_type_conversion
    def build(self, input_shape):
        self.drop = layers.Dropout(self.dropout)
        self.proj = HeadProjection(self.classes, 1)
        self.act = HeadActivation(self.classes)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        predictions, images = inputs

        outputs = self.drop(predictions)
        outputs = self.proj(outputs)
        outputs = resize_by_sample([outputs, images])

        outputs = self.act(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[1][:-1] + (self.classes,)

    def compute_output_signature(self, input_signature):
        shape = self.compute_output_shape([i.shape for i in input_signature])

        return tf.TensorSpec(dtype='float32', shape=shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'classes': self.classes,
            'dropout': self.dropout
        })

        return config
