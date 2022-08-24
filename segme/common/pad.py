import tensorflow as tf
import warnings
from keras import backend, layers
from keras.utils.conv_utils import normalize_tuple
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='SegMe>Common')
class SymmetricPadding(layers.ZeroPadding2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if max(self.padding[0] + self.padding[1]) > 1:
            raise ValueError('Symmetric padding can lead to misbehavior when padding size > 1')

    def call(self, inputs):
        padding = self.padding
        data_format = self.data_format

        assert len(padding) == 2
        assert len(padding[0]) == 2
        assert len(padding[1]) == 2

        if data_format is None:
            data_format = backend.image_data_format()
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format: ' + str(data_format))

        if data_format == 'channels_first':
            pattern = [[0, 0], [0, 0], list(padding[0]), list(padding[1])]
        else:
            pattern = [[0, 0], list(padding[0]), list(padding[1]), [0, 0]]

        return tf.pad(inputs, pattern, mode='SYMMETRIC')


@register_keras_serializable(package='SegMe>Common')
class SamePadding(layers.Layer):
    def __init__(self, kernel_size, strides, dilation_rate, symmetric_pad=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.kernel_size = normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = normalize_tuple(strides, 2, 'strides', allow_zero=True)
        self.dilation_rate = normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.symmetric_pad = symmetric_pad

        self.apply_pad = (1, 1) != self.kernel_size and (
                (1, 1) != self.strides or (1, 1) != self.dilation_rate or self.symmetric_pad)
        if self.apply_pad:
            warnings.warn(
                f'Required "same" padding (kernel_size: {self.kernel_size}, strides: {self.strides}, dilation_rate: '
                f'{self.dilation_rate}, symmetric_pad: {self.symmetric_pad}) will create a copy of input tensor')
            pad_h = self.dilation_rate[0] * (self.kernel_size[0] - 1)
            pad_w = self.dilation_rate[1] * (self.kernel_size[1] - 1)
            self._paddings = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))

        self.padding_used = False

    @property
    def padding(self):
        self.padding_used = True

        return 'valid' if self.apply_pad else 'same'

    @shape_type_conversion
    def build(self, input_shape):
        if self.apply_pad:
            symmetric_pad = self.symmetric_pad is True or \
                            self.symmetric_pad is None and 1 == max(self._paddings[0] + self._paddings[1])
            if symmetric_pad:
                self.pad = SymmetricPadding(self._paddings)
            else:
                self.pad = layers.ZeroPadding2D(self._paddings)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if not self.padding_used:
            warnings.warn('Padding layer called without reading padding mode property')

        if not self.apply_pad:
            return inputs

        return self.pad(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if not self.apply_pad:
            return input_shape

        return self.pad.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'dilation_rate': self.dilation_rate,
            'symmetric_pad': self.symmetric_pad
        })

        return config
