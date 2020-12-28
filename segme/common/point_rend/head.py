import tensorflow as tf
from tensorflow.keras import initializers, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe>PointRend')
class PointHead(layers.Layer):
    def __init__(self, classes, units, fines, residual=False, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=3, axes={-1: classes})]  # coarse features
        self.input_spec += [layers.InputSpec(ndim=3) for _ in range(fines)]  # fine grained features

        if fines < 1:
            raise ValueError('At least one fine grained feature map required')

        self.classes = classes
        self.units = units
        self.fines = fines
        self.residual = residual

    @shape_type_conversion
    def build(self, input_shape):
        # Official implementation initializes with
        # nn.init.kaiming_normal_(..., mode='fan_out', nonlinearity='relu')
        # https://github.com/facebookresearch/detectron2/blob/master/projects/PointRend/point_rend/point_head.py#L137
        weight_init = initializers.he_normal()
        layers_kwargs = {'padding': 'same', 'activation': 'relu', 'kernel_initializer': weight_init}
        self.layers = [layers.Conv1D(u, 1, **layers_kwargs) for u in self.units]

        proj_init = initializers.random_normal(stddev=0.001)
        self.proj = layers.Conv1D(self.classes, 1, padding='same', kernel_initializer=proj_init)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = layers.concatenate(inputs)  # coarse + fine

        for layer in self.layers:
            outputs = layer(outputs)
            if self.residual:
                outputs = layers.concatenate([inputs[0], outputs])  # coarse + intermediate

        outputs = self.proj(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.classes,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'classes': self.classes,
            'units': self.units,
            'fines': self.fines,
            'residual': self.residual
        })

        return config
