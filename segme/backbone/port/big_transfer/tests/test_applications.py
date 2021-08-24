from absl.testing import parameterized
from keras import backend
from tensorflow.python.platform import test
from ..bit import BiT_S_R50x1, BiT_S_R50x3, BiT_S_R101x1, BiT_S_R101x3, BiT_S_R152x4
from ..bit import BiT_M_R50x1, BiT_M_R50x3, BiT_M_R101x1, BiT_M_R101x3, BiT_M_R152x4

MODEL_LIST = [
    (BiT_S_R50x1, 2048),
    # Bad weights
    # (BiT_S_R50x3, 2048),
    # (BiT_S_R101x1, 2048),
    (BiT_S_R101x3, 6144),
    (BiT_S_R152x4, 8192),
    (BiT_M_R50x1, 2048),
    (BiT_M_R50x3, 6144),
    (BiT_M_R101x1, 2048),
    (BiT_M_R101x3, 6144),
    (BiT_M_R152x4, 8192)
]


class ApplicationsTest(test.TestCase, parameterized.TestCase):
    @parameterized.parameters(*MODEL_LIST)
    def test_application_base(self, app, _):
        # Can be instantiated with default arguments
        model = app(weights=None)
        # Can be serialized and deserialized
        config = model.get_config()
        reconstructed_model = model.__class__.from_config(config)
        self.assertEqual(len(model.weights), len(reconstructed_model.weights))
        backend.clear_session()

    @parameterized.parameters(*MODEL_LIST)
    def test_application_notop(self, app, last_dim):
        output_shape = _get_output_shape(lambda: app(weights=None, include_top=False))
        _assert_shape_equal(output_shape, (None, None, None, last_dim))
        backend.clear_session()

    @parameterized.parameters(*MODEL_LIST)
    def test_application_variable_input_channels(self, app, last_dim):
        if backend.image_data_format() == 'channels_first':
            input_shape = (1, None, None)
        else:
            input_shape = (None, None, 1)
        output_shape = _get_output_shape(lambda: app(weights=None, include_top=False, input_shape=input_shape))
        _assert_shape_equal(output_shape, (None, None, None, last_dim))
        backend.clear_session()

        if backend.image_data_format() == 'channels_first':
            input_shape = (4, None, None)
        else:
            input_shape = (None, None, 4)
        output_shape = _get_output_shape(lambda: app(weights=None, include_top=False, input_shape=input_shape))
        _assert_shape_equal(output_shape, (None, None, None, last_dim))
        backend.clear_session()


def _assert_shape_equal(shape1, shape2):
    if len(shape1) != len(shape2):
        raise AssertionError('Shapes are different rank: %s vs %s' % (shape1, shape2))
    for v1, v2 in zip(shape1, shape2):
        if v1 != v2:
            raise AssertionError('Shapes differ: %s vs %s' % (shape1, shape2))


def _get_output_shape(model_fn):
    model = model_fn()

    return model.output_shape


if __name__ == '__main__':
    test.main()
