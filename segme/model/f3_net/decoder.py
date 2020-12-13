from tensorflow.keras import Sequential, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from .cfm import CFM
from ...common import resize_by_sample


@utils.register_keras_serializable(package='SegMe>F3Net')
class Decoder(layers.Layer):
    def __init__(self, refine, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4) for _ in range(4 + int(refine))]
        self.refine = refine
        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        self.cfm45 = CFM(self.filters)
        self.cfm34 = CFM(self.filters)
        self.cfm23 = CFM(self.filters)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        out2h, out3h, out4h, out5v = inputs[:4]

        if self.refine:
            fback = inputs[4]
            refine5 = resize_by_sample([fback, out5v])
            refine4 = resize_by_sample([fback, out4h])
            refine3 = resize_by_sample([fback, out3h])
            refine2 = resize_by_sample([fback, out2h])
            out5v = layers.add([out5v, refine5])
            out4h, out4v = self.cfm45([layers.add([out4h, refine4]), out5v])
            out3h, out3v = self.cfm34([layers.add([out3h, refine3]), out4v])
            out2h, pred = self.cfm23([layers.add([out2h, refine2]), out3v])
        else:
            out4h, out4v = self.cfm45([out4h, out5v])
            out3h, out3v = self.cfm34([out3h, out4v])
            out2h, pred = self.cfm23([out2h, out3v])

        return out2h, out3h, out4h, out5v, pred

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        out2h_shape, out3h_shape, out4h_shape, out5v_shape = input_shape[:4]

        return [out2h_shape[:-1] + (self.filters,), out3h_shape[:-1] + (self.filters,),
                out4h_shape[:-1] + (self.filters,), out5v_shape, out2h_shape[:-1] + (self.filters,)]

    def get_config(self):
        config = super().get_config()
        config.update({
            'refine': self.refine,
            'filters': self.filters
        })

        return config
