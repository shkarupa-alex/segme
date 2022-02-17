import tensorflow as tf
from keras import backend, layers
from keras.utils.generic_utils import register_keras_serializable


@register_keras_serializable(package='SegMe')
class StandardizedConv2D(layers.Conv2D):
    """Implements the abs/1903.10520 technique (see go/dune-gn).

    You can simply replace any Conv2D with this one to use re-parametrized
    convolution operation in which the kernels are standardized before conv.

    Note that it does not come with extra learnable scale/bias parameters,
    as those used in "Weight normalization" (abs/1602.07868). This does not
    matter if combined with BN/GN/..., but it would matter if the convolution
    was used standalone.

    Author: Lucas Beyer
    """

    def convolution_op(self, inputs, kernel):
        # Kernel has shape HWIO, normalize over HWI
        mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)

        # Author code uses std + 1e-5
        kernel_ = (kernel - mean) / tf.sqrt(var + 1e-10)

        return super().convolution_op(inputs, kernel_)


@register_keras_serializable(package='SegMe')
class StandardizedDepthwiseConv2D(layers.DepthwiseConv2D):
    """Implements the abs/1903.10520 technique for DepthwiseConv2D."""

    def call(self, inputs):
        kernel = self.depthwise_kernel

        # Kernel has shape HWIO, normalize over HWI
        mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)

        # Author code uses std + 1e-5
        kernel_ = (kernel - mean) / tf.sqrt(var + 1e-10)

        outputs = backend.depthwise_conv2d(
            inputs, kernel_, strides=self.strides, padding=self.padding, dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.use_bias:
            outputs = backend.bias_add(outputs, self.bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs
