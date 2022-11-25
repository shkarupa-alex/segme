from keras import backend, layers, models
from keras.utils.control_flow_util import smart_cond
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvNormAct
from segme.common.interrough import BilinearInterpolation


@register_keras_serializable(package='SegMe>Common')
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
            ConvNormAct(self.filters, 3, name='attention_cna0'),
            ConvNormAct(self.filters, 3, name='attention_cna1'),
            layers.Dropout(self.dropout, name='attention_drop'),
            layers.Conv2D(1, 1, activation='sigmoid', use_bias=False, name='attention_proj')
        ], name='attention')

        self.intbyscale = {str(scale): BilinearInterpolation(scale)
                           for scale in set(self.train_scales + self.eval_scales)}
        self.intbysample = BilinearInterpolation(None)

        super().build(input_shape)

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

        for scale in scales:  # TODO: check order
            _inputs = self.intbyscale[str(scale)](inputs)  # TODO: bicubic? +antialiasing?
            _outputs, _features = self.layer.call(_inputs)
            _outputs = self.intbysample([_outputs, _inputs])

            if outputs is None:
                # store largest
                outputs = _outputs
                continue

            _attention = self.attention(_features)
            _attention = self.intbysample([_attention, _inputs])

            if scale >= 1.0:
                # downscale previous
                outputs = self.intbysample([outputs, _outputs])
                outputs = _attention * _outputs + (1. - _attention) * outputs
            else:
                # upscale current
                _outputs = _attention * _outputs
                _outputs = self.intbysample([_outputs, outputs])
                _attention = self.intbysample([_attention, outputs])

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
