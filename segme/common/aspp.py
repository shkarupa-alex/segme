from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.convnormact import ConvNormAct
from segme.common.resize import NearestInterpolation
from segme.common.sequence import Sequence


@register_keras_serializable(package="SegMe>Common")
class AtrousSpatialPyramidPooling(layers.Layer):
    _stride_rates = {8: [12, 24, 36], 16: [6, 12, 18], 32: [3, 6, 9]}

    def __init__(self, filters, stride, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        self.filters = filters
        self.stride = stride
        self.dropout = dropout

        if stride not in self._stride_rates:
            raise NotImplementedError("Unsupported input stride")

    def build(self, input_shape):
        self.conv1 = ConvNormAct(self.filters, 1, dtype=self.dtype_policy)
        self.conv1.build(input_shape)

        rate0, rate1, rate2 = self._stride_rates[self.stride]
        self.conv3r0 = Sequence(
            [
                ConvNormAct(
                    None,
                    3,
                    dilation_rate=rate0,
                    name="dna",
                    dtype=self.dtype_policy,
                ),
                ConvNormAct(
                    self.filters, 1, name="pna", dtype=self.dtype_policy
                ),
            ],
            name="conv3r0",
            dtype=self.dtype_policy,
        )
        self.conv3r0.build(input_shape)

        self.conv3r1 = Sequence(
            [
                ConvNormAct(
                    None,
                    3,
                    dilation_rate=rate1,
                    name="dna",
                    dtype=self.dtype_policy,
                ),
                ConvNormAct(
                    self.filters, 1, name="pna", dtype=self.dtype_policy
                ),
            ],
            name="conv3r1",
            dtype=self.dtype_policy,
        )
        self.conv3r1.build(input_shape)

        self.conv3r2 = Sequence(
            [
                ConvNormAct(
                    None,
                    3,
                    dilation_rate=rate2,
                    name="dna",
                    dtype=self.dtype_policy,
                ),
                ConvNormAct(
                    self.filters, 1, name="pna", dtype=self.dtype_policy
                ),
            ],
            name="conv3r2",
            dtype=self.dtype_policy,
        )
        self.conv3r2.build(input_shape)

        self.pool = Sequence(
            [
                layers.GlobalAveragePooling2D(
                    keepdims=True, dtype=self.dtype_policy
                ),
                ConvNormAct(
                    self.filters, 1, name="cna", dtype=self.dtype_policy
                ),
            ],
            name="pool",
            dtype=self.dtype_policy,
        )
        self.pool.build(input_shape)

        self.proj = Sequence(
            [
                ConvNormAct(
                    self.filters, 1, name="cna", dtype=self.dtype_policy
                ),
                layers.Dropout(
                    self.dropout, name="drop", dtype=self.dtype_policy
                ),
            ],
            name="pool",
            dtype=self.dtype_policy,
        )
        self.proj.build(input_shape[:-1] + (self.filters * 5,))

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = [
            self.conv1(inputs),
            self.conv3r0(inputs),
            self.conv3r1(inputs),
            self.conv3r2(inputs),
        ]

        pool = self.pool(inputs)
        pool = NearestInterpolation(None, dtype=self.dtype_policy)(
            [pool, inputs]
        )
        outputs.append(pool)

        outputs = ops.concatenate(outputs, axis=-1)
        outputs = self.proj(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "stride": self.stride,
                "dropout": self.dropout,
            }
        )

        return config
