import numpy as np
from keras import Model, layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ...backbone import Backbone
from ...backbone.utils import patch_config


@register_keras_serializable(package='SegMe>FBAMatting')
class Encoder(layers.Layer):
    def __init__(self, bone_arch, bone_init, **kwargs):
        super().__init__(**kwargs)
        self.input_channels = 11
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: self.input_channels})
        self.bone_arch = bone_arch
        self.bone_init = bone_init

    @shape_type_conversion
    def build(self, input_shape):
        base_model = Backbone(self.bone_arch, self.bone_init, trainable=True, scales=[2, 4, 8])
        base_model.build(input_shape[:-1] + (3,))

        zeros_shape = (7, 7, self.input_channels - 3, 64)
        ext_weights = [w if w.shape != (7, 7, 3, 64) else np.concatenate([w, np.zeros(zeros_shape)], axis=2)
                       for w in base_model.get_weights()]

        ext_model = Backbone(self.bone_arch, None, trainable=True, scales=[2, 4, 8])
        ext_model.build(input_shape)
        ext_model.set_weights(ext_weights)

        self.backbone = ext_model

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.backbone(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.backbone.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'bone_arch': self.bone_arch,
            'bone_init': self.bone_init,
        })

        return config
