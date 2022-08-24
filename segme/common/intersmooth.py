from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.policy import respol


@register_keras_serializable(package='SegMe>Common')
class SmoothInterpolation(layers.Layer):
    def __init__(self, scale, resize_type=True, resize_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4) if scale is not None else [
            layers.InputSpec(ndim=4), layers.InputSpec(ndim=4)]  # targets, samples

        self.scale = None if scale is None else float(scale)

        self.resize_kwargs = resize_kwargs
        if self.resize_kwargs is not None and not isinstance(self.resize_kwargs, dict):
            raise ValueError('Resize kwargs must be a dict if provided')

        policy = respol.global_policy()
        self.resize_type = False
        if isinstance(resize_type, str) and resize_type:
            self.resize_type = resize_type
        elif resize_type is True:
            self.resize_type = policy.name
        elif bool(resize_type):
            raise ValueError('Unknown resize type')
        if not self.resize_type:
            raise ValueError('Resize type can\'t be empty')

    @shape_type_conversion
    def build(self, input_shape):
        resize_kwargs = self.resize_kwargs or {}
        self.resize = respol.RESIZERS.new(self.resize_type, self.scale, **resize_kwargs)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.resize(inputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.resize.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'scale': self.scale,
            'resize_type': self.resize_type,
            'resize_kwargs': self.resize_kwargs
        })

        return config
