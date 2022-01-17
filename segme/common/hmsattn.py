from keras import backend, layers, models
from keras.utils.control_flow_util import smart_cond
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .convbnrelu import ConvBnRelu
from .resizebysample import resize_by_sample
from .resizebyscale import resize_by_scale


@register_keras_serializable(package='Miss')
class HierarchicalMultiScaleAttention(layers.Wrapper):
    """ Proposed in: https://arxiv.org/abs/2005.10821

    Arguments:
      layer: The `Layer` instance to be wrapped.
    """

    def __init__(self, layer, scales=((0.5,), (0.25, 0.5, 2.0)), filters=256, dropout=0., **kwargs):
        super().__init__(layer, **kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.scales = scales
        self.filters = filters
        self.dropout = dropout

        if 2 != len(scales) or not all([isinstance(s, (list, tuple)) for s in scales]):
            raise ValueError('Expecting `scales` to be a train/eval pair of scale lists/tuples.')

        self.train_scales = sorted({1.0} | set(scales[0]), reverse=True)
        self.eval_scales = sorted({1.0} | set(scales[1]), reverse=True)

        if len(self.train_scales) < 2 or len(self.eval_scales) < 2:
            raise ValueError('Expecting `scales` to have at least one more scale except `1`.')

    def build(self, input_shape=None):
        self.attention = models.Sequential([
            ConvBnRelu(self.filters, 3),
            ConvBnRelu(self.filters, 3),
            layers.Dropout(self.dropout),
            layers.Conv2D(1, kernel_size=1, padding='same', use_bias=False, activation='sigmoid')
        ])

        super().build(input_shape)

        layer_shape = self.layer.compute_output_shape(input_shape)
        if not isinstance(layer_shape, (list, tuple)) or 2 != len(layer_shape):
            raise ValueError('Expecting `layer` to return 2 outputs: logits and high-level features.')
        if {4} != set([len(s) for s in layer_shape]):
            raise ValueError('Expecting `layer` to return 2 4D outputs: logits and high-level features.')

    def call(self, inputs, training=None, **kwargs):
        if training is None:
            training = backend.learning_phase()

        outputs = smart_cond(
            training,
            lambda: self._branch(inputs, self.train_scales),
            lambda: self._branch(inputs, self.eval_scales))

        return outputs

    def _branch(self, inputs, scales):
        outputs = None

        for scale in scales:
            _inputs = resize_by_scale(inputs, scale=scale)
            _outputs, _features = self.layer.call(_inputs)
            _outputs = resize_by_sample([_outputs, _inputs])

            if outputs is None:
                # store largest
                outputs = _outputs
                continue

            _attention = self.attention(_features)
            _attention = resize_by_sample([_attention, _inputs])

            if scale >= 1.0:
                # downscale previous
                outputs = resize_by_sample([outputs, _outputs])
                outputs = _attention * _outputs + (1. - _attention) * outputs
            else:
                # upscale current
                _outputs = _attention * _outputs
                _outputs = resize_by_sample([_outputs, outputs])
                _attention = resize_by_sample([_attention, outputs])

                outputs = _outputs + (1 - _attention) * outputs

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + self.layer.compute_output_shape(input_shape)[0][-1:]

    def get_config(self):
        config = super().get_config()
        config.update({
            'scales': self.scales,
            'filters': self.filters,
            'dropout': self.dropout
        })

        return config
