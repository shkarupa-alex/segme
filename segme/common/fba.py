import tensorflow as tf
from keras import backend, layers
from keras.saving import register_keras_serializable
from keras.src.utils.control_flow_util import smart_cond
from keras.src.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='SegMe>Common')
class FBAFusion(layers.Layer):
    def __init__(self, inference_only=True, alpha_variance=10., num_repeats=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}),  # image
            layers.InputSpec(ndim=4, axes={-1: 3}),  # fg
            layers.InputSpec(ndim=4, axes={-1: 3}),  # bg
            layers.InputSpec(ndim=4, axes={-1: 1})]  # alpha

        self.inference_only = inference_only
        self.alpha_variance = alpha_variance
        self.num_repeats = num_repeats

    def call(self, inputs, training=None, *args, **kwargs):
        if not self.inference_only:
            return self.fuse(inputs)

        if training is None:
            training = backend.learning_phase()

        return smart_cond(training, lambda: inputs[1:], lambda: self.fuse(inputs))

    def fuse(self, inputs):
        img, fg, bg, alpha = inputs

        for _ in range(self.num_repeats):
            fg, bg, alpha = self.fuse_step(img, fg, bg, alpha)

        return fg, bg, alpha

    def fuse_step(self, img, fg, bg, alpha):
        # TODO: use real variancies?
        var_fg = var_bg = var_img = 1.
        var_alpha = self.alpha_variance

        inv_alpha = 1. - alpha
        img_delpta = (img - alpha * fg - inv_alpha * bg)
        fg_upd = fg + (var_fg / var_img) * alpha * img_delpta
        bg_upd = bg + (var_bg / var_img) * inv_alpha * img_delpta

        fgbg_delta = fg_upd - bg_upd
        alpha_upd = alpha + tf.reduce_sum((var_alpha / var_img) * (img - bg_upd) * fgbg_delta, axis=-1, keepdims=True)
        alpha_upd /= 1. + tf.reduce_sum((var_alpha / var_img) * tf.square(fgbg_delta), axis=-1, keepdims=True)

        return fg_upd, bg_upd, alpha_upd

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[1:]

    def get_config(self):
        config = super().get_config()
        config.update({
            'inference_only': self.inference_only,
            'alpha_variance': self.alpha_variance,
            'num_repeats': self.num_repeats
        })

        return config
