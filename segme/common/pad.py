import tensorflow as tf
from keras import backend, layers
from keras.utils.generic_utils import register_keras_serializable


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
