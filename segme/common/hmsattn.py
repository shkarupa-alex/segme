from keras.src import backend
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable
# from keras.src.utils.control_flow_util import ops.cond

from segme.common.convnormact import ConvNormAct
from segme.common.resize import BilinearInterpolation
from segme.common.sequence import Sequence


@register_keras_serializable(package="SegMe>Common")
class HierarchicalMultiScaleAttention(layers.Wrapper):
    """Proposed in: https://arxiv.org/abs/2005.10821

    Arguments:
      layer: The `Layer` instance to be wrapped.
    """

    def __init__(
        self,
        layer,
        scales=((0.5,), (0.25, 0.5, 2.0)),
        filters=256,
        dropout=0.0,
        **kwargs
    ):
        super().__init__(layer, **kwargs)
        self.input_spec = InputSpec(ndim=4)
        self.scales = scales
        self.filters = filters
        self.dropout = dropout

        if 2 != len(scales) or not all(
            [isinstance(s, (list, tuple)) for s in scales]
        ):
            raise ValueError(
                "Expecting `scales` to be a train/eval pair of scale lists/tuples."
            )

        self.train_scales = sorted({1.0} | set(scales[0]), reverse=True)
        self.eval_scales = sorted({1.0} | set(scales[1]), reverse=True)

        if len(self.train_scales) < 2 or len(self.eval_scales) < 2:
            raise ValueError(
                "Expecting `scales` to have at least one more scale except `1`."
            )

    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        current_shape = self.layer.compute_output_shape(input_shape)

        self.attention = Sequence(
            [
                ConvNormAct(self.filters, 3, name="cna0", dtype=self.dtype_policy),
                ConvNormAct(self.filters, 3, name="cna1", dtype=self.dtype_policy),
                layers.Dropout(self.dropout, name="drop", dtype=self.dtype_policy),
                layers.Conv2D(
                    1, 1, activation="sigmoid", use_bias=False, name="proj", dtype=self.dtype_policy
                ),
            ],
            name="attention", dtype=self.dtype_policy
        )
        self.attention.build(current_shape[1])

        self.intbyscale = {
            str(scale): BilinearInterpolation(scale, dtype=self.dtype_policy)
            for scale in set(self.train_scales + self.eval_scales)
        }
        self.intbysample = BilinearInterpolation(None, dtype=self.dtype_policy)

        super().build(input_shape)

    def call(self, inputs, training=False, **kwargs):
        return self._branch(inputs, self.train_scales if training else self.eval_scales)

    def _branch(self, inputs, scales):
        outputs = None

        for scale in scales:  # TODO: check order
            _inputs = self.intbyscale[str(scale)](
                inputs
            )  # TODO: bicubic? +antialiasing?
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
                outputs = _attention * _outputs + (1.0 - _attention) * outputs
            else:
                # upscale current
                _outputs = _attention * _outputs
                _outputs = self.intbysample([_outputs, outputs])
                _attention = self.intbysample([_attention, outputs])

                outputs = _outputs + (1 - _attention) * outputs

        return outputs

    def compute_output_shape(self, input_shape):
        return (
            input_shape[:-1]
            + self.layer.compute_output_shape(input_shape)[0][-1:]
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "scales": self.scales,
                "filters": self.filters,
                "dropout": self.dropout,
            }
        )

        return config
