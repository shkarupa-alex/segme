import tensorflow as tf
from keras import layers
from keras.utils.conv_utils import normalize_tuple
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tensorflow.python.platform.device_context import enclosing_tpu_context
from segme.common.convnormact import Conv, Norm, Act, ConvNormAct


@register_keras_serializable(package='SegMe>Common')
class FourierUnit(layers.Layer):
    def __init__(self, filters, **kwargs):
        kwargs['autocast'] = False
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        nchw_support = tf.test.is_gpu_available() or enclosing_tpu_context() is not None
        self.data_format = 'channels_first' if nchw_support else 'channels_last'
        self.cna = ConvNormAct(self.filters * 2, 1, data_format=self.data_format, dtype='float32', name='cna')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        batch, height, width, _ = tf.unstack(tf.shape(inputs))
        ortho_norm = tf.math.sqrt(tf.cast(height * width, 'float32'))

        # to RDFT
        outputs = tf.cast(inputs, 'float32')
        outputs = tf.transpose(outputs, [0, 3, 1, 2])
        outputs = tf.signal.rfft2d(outputs / ortho_norm)
        outputs = tf.concat([tf.math.real(outputs), tf.math.imag(outputs)], axis=1)

        outputs.set_shape(outputs.shape[:1] + (self.channels * 2,) + outputs.shape[2:])

        if 'channels_last' == self.data_format:
            outputs = tf.transpose(outputs, [0, 2, 3, 1])

        outputs = self.cna(outputs)

        if 'channels_last' == self.data_format:
            outputs = tf.transpose(outputs, [0, 3, 1, 2])

        # from RDFT
        outputs = tf.dtypes.complex(*tf.split(outputs, 2, axis=1))
        outputs = tf.signal.irfft2d(outputs, fft_length=[height, width]) * ortho_norm
        outputs = tf.transpose(outputs, [0, 2, 3, 1])
        outputs = tf.cast(outputs, inputs.dtype)
        outputs.set_shape(inputs.shape[:-1] + (self.filters,))

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config


@register_keras_serializable(package='SegMe>Common')
class SpectralTransform(layers.Layer):
    def __init__(self, filters, strides=(1, 1), use_bias=False, use_lfu=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.strides = normalize_tuple(strides, 2, 'strides')
        self.use_bias = use_bias
        self.use_lfu = use_lfu

        if self.filters // 2 // 4 < 1 and self.use_lfu:
            raise ValueError('Too few filters to use local fourier unit.')

    @shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.downsample = layers.Activation('linear', name='downsample') if 1 == max(self.strides) \
            else layers.AvgPool2D(self.strides, padding='same', name='downsample')
        self.reduce = ConvNormAct(self.filters // 2, 1, name='reduce')
        self.gfu = FourierUnit(self.filters // 2, name='gfu')
        if self.use_lfu:
            self.lfu = FourierUnit(self.filters // 2, name='lfu')
        self.proj = layers.Conv2D(self.filters, 1, use_bias=self.use_bias, name='proj')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = self.downsample(inputs)
        outputs = self.reduce(outputs)

        gfu = self.gfu(outputs)

        if self.use_lfu:
            batch, height, width, _ = tf.unstack(tf.shape(outputs))

            assert_height = tf.debugging.assert_equal(height % 2, 0)
            assert_width = tf.debugging.assert_equal(width % 2, 0)
            with tf.control_dependencies([assert_height, assert_width]):
                height2, width2, channel4 = height // 2, width // 2, self.channels // 4
                lfu = outputs[..., :channel4]

            lfu = tf.reshape(lfu, [batch, 2, height2, 2, width2, channel4])
            lfu = tf.transpose(lfu, [0, 2, 4, 3, 1, 5])
            lfu = tf.reshape(lfu, [batch, height2, width2, channel4 * 4])
            lfu = self.lfu(lfu)
            lfu = tf.tile(lfu, [1, 2, 2, 1])

            outputs += lfu

        outputs = self.proj(outputs + gfu)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = self.downsample.compute_output_shape(input_shape)

        return output_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'strides': self.strides,
            'use_bias': self.use_bias,
            'use_lfu': self.use_lfu
        })

        return config


@register_keras_serializable(package='SegMe>Common')
class FastFourierConv(layers.Layer):
    """ Proposed in: https://papers.nips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf """

    def __init__(self, filters, kernel_size, ratio, strides=(1, 1), dilation_rate=(1, 1), use_bias=False, use_lfu=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.ratio = ratio
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.use_lfu = use_lfu

        if not 0. <= self.ratio <= 1.:
            raise ValueError('Output global/local ratio must be in range [0; 1].')

    @shape_type_conversion
    def build(self, input_shape):
        if 2 == len(input_shape):
            self.channels = [shape[-1] for shape in input_shape]
            self.input_spec = [layers.InputSpec(ndim=4, axes={-1: c}) for c in self.channels]
        else:
            self.channels = [input_shape[-1], 0]
            self.input_spec = layers.InputSpec(ndim=4, axes={-1: self.channels[0]})
        if None in self.channels:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.filters_global = int(self.filters * self.ratio)
        self.filters_local = self.filters - self.filters_global
        if 0 == self.filters_local:
            raise ValueError('Local branch filters must be greater then 0.')

        self.l2l = Conv(self.filters_local, self.kernel_size, strides=self.strides, padding='same',
                        dilation_rate=self.dilation_rate, use_bias=self.use_bias, name='l2l')
        self.g2l = None if 0 == self.channels[1] \
            else Conv(self.filters_local, self.kernel_size, strides=self.strides, padding='same',
                      dilation_rate=self.dilation_rate, use_bias=self.use_bias, name='g2l')
        self.l2g = None if 0 == self.filters_global \
            else Conv(self.filters_global, self.kernel_size, strides=self.strides, padding='same',
                      dilation_rate=self.dilation_rate, use_bias=self.use_bias, name='l2g')
        self.g2g = None if 0 == self.channels[1] or 0 == self.filters_global \
            else SpectralTransform(self.filters_global, strides=self.strides, use_bias=self.use_bias,
                                   use_lfu=self.use_lfu, name='g2g')

        self.norm_global = None if 0 == self.filters_global else Norm(name='norm_global')
        self.norm_local = None if 0 == self.filters_local else Norm(name='norm_local')
        self.act = Act(name='act')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        if 0 == self.channels[1]:
            inputs_local, inputs_global = inputs, None
        else:
            inputs_local, inputs_global = inputs

        outputs_local, outputs_global = [], []
        outputs_local.append(self.l2l(inputs_local))
        if 0 != self.channels[1]:
            outputs_local.append(self.g2l(inputs_global))
        if 0 != self.filters_global:
            outputs_global.append(self.l2g(inputs_local))
        if 0 != self.channels[1] and 0 != self.filters_global:
            outputs_global.append(self.g2g(inputs_global))

        if 0 != self.filters_local:
            outputs_local = sum(outputs_local)
            outputs_local = self.norm_local(outputs_local)
            outputs_local = self.act(outputs_local)
        if 0 != self.filters_global:
            outputs_global = sum(outputs_global)
            outputs_global = self.norm_global(outputs_global)
            outputs_global = self.act(outputs_global)

        if 0 == self.filters_global:
            return outputs_local

        if 0 == self.filters_local:
            return outputs_global

        return outputs_local, outputs_global

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0] if 2 == len(input_shape) else input_shape
        output_shape = self.l2l.compute_output_shape(output_shape) if self.l2l is not None \
            else self.l2g.compute_output_shape(output_shape)

        if 0 == self.filters_local or 0 == self.filters_global:
            return output_shape[:-1] + (self.filters,)

        return output_shape[:-1] + (self.filters_local,), output_shape[:-1] + (self.filters_global,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'ratio': self.ratio,
            'strides': self.strides,
            'dilation_rate': self.dilation_rate,
            'use_bias': self.use_bias,
            'use_lfu': self.use_lfu
        })

        return config
