import tensorflow as tf
from keras import backend, layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .convnormrelu import ConvNormRelu
from .sameconv import SameConv


@register_keras_serializable(package='SegMe')
class BoxFilter(layers.Layer):
    def __init__(self, radius, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.radius = radius

    def call(self, inputs, **kwargs):
        outputs = tf.cumsum(inputs, axis=1)

        left = outputs[:, self.radius:2 * self.radius + 1]
        middle = outputs[:, 2 * self.radius + 1:] - outputs[:, :-2 * self.radius - 1]
        right = outputs[:, -1:] - outputs[:, -2 * self.radius - 1:-self.radius - 1]
        outputs = tf.concat([left, middle, right], axis=1)

        outputs = tf.cumsum(outputs, axis=2)

        left = outputs[:, :, self.radius:2 * self.radius + 1]
        middle = outputs[:, :, 2 * self.radius + 1:] - outputs[:, :, :-2 * self.radius - 1]
        right = outputs[:, :, -1:] - outputs[:, :, -2 * self.radius - 1:-self.radius - 1]

        outputs = tf.concat([left, middle, right], axis=2)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'radius': self.radius})

        return config


# @register_keras_serializable(package='SegMe')
# class FastGuidedFilter(nn.Module):
#     def __init__(self, r, eps=1e-8):
#         super(FastGuidedFilter, self).__init__()
#
#         self.r = r
#         self.eps = eps
#         self.boxfilter = BoxFilter(r)
#
#
#     def forward(self, lr_x, lr_y, hr_x):
#         n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
#         n_lry, c_lry, h_lry, w_lry = lr_y.size()
#         n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()
#
#         assert n_lrx == n_lry and n_lry == n_hrx
#         assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
#         assert h_lrx == h_lry and w_lrx == w_lry
#         assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1
#
#         ## N
#         N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))
#
#         ## mean_x
#         mean_x = self.boxfilter(lr_x) / N
#         ## mean_y
#         mean_y = self.boxfilter(lr_y) / N
#         ## cov_xy
#         cov_xy = self.boxfilter(lr_x * lr_y) / N - mean_x * mean_y
#         ## var_x
#         var_x = self.boxfilter(lr_x * lr_x) / N - mean_x * mean_x
#
#         ## A
#         A = cov_xy / (var_x + self.eps)
#         ## b
#         b = mean_y - A * mean_x
#
#         ## mean_A; mean_b
#         mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
#         mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
#
#         return mean_A*hr_x+mean_b

@register_keras_serializable(package='SegMe')
class GuidedFilter(layers.Layer):
    def __init__(self, radius, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4),  # guide
            layers.InputSpec(ndim=4)  # target
        ]

        self.radius = radius

    @shape_type_conversion
    def build(self, input_shape):
        self.box = BoxFilter(self.radius)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        guides, targets = inputs

        guides_batch, guides_height, guides_width, guides_channel = tf.unstack(tf.shape(guides))
        targets_batch, targets_height, targets_width, targets_channel = tf.unstack(tf.shape(targets))
        
        assert_batch = tf.assert_equal(guides_batch, targets_batch)
        assert_height = tf.assert_equal(guides_height, targets_height)
        assert_width = tf.assert_equal(guides_width, targets_width)
        assert_hradius = tf.assert_greater(guides_height, 2 * self.radius + 1)
        assert_wradius = tf.assert_greater(guides_width, 2 * self.radius + 1)
        assert_channel = tf.assert_equal((guides_channel == 1) | (guides_channel == targets_channel), True)

        with tf.control_dependencies([
            assert_batch, assert_height, assert_width, assert_hradius, assert_wradius, assert_channel]):

            size = tf.ones((1, guides_height, guides_width, 1), dtype=self.compute_dtype)
            size = self.box(size)

            guides_mean = self.box(guides) / size
            targets_mean = self.box(targets) / size

            covariance = self.box(guides * targets) / size - guides_mean * targets_mean
            variance = self.box(guides ** 2) / size - guides_mean ** 2

            scale = covariance / (variance + backend.epsilon())
            bias = targets_mean - scale * guides_mean

            scale_mean = self.box(scale) / size
            scale_bias = self.box(bias) / size

            outputs = scale_mean * guides + scale_bias

            return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def get_config(self):
        config = super().get_config()
        config.update({'radius': self.radius})

        return config

#
#
# class ConvGuidedFilter(nn.Module):
#     def __init__(self, radius=1, norm=nn.BatchNorm2d):
#         super(ConvGuidedFilter, self).__init__()
#
#         self.box_filter = layers.Conv2D(3, 3, padding=radius, dilation_rate=radius, groups=3, use_bias=False)
#         # TODO
#         # self.box_filter = nn.Conv2d(3, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=3)
#         self.conv_a = models.Sequential([
#             ConvNormRelu(32, 1),
#             ConvNormRelu(32, 1),
#             SameConv(3, 1, use_bias=False)
#         ])
#         self.box_filter.weight.data[...] = 1.0
#
#     def forward(self, x_lr, y_lr, x_hr):
#         _, _, h_lrx, w_lrx = x_lr.size()
#         _, _, h_hrx, w_hrx = x_hr.size()
#
#         N = self.box_filter(x_lr.data.new().resize_((1, 3, h_lrx, w_lrx)).fill_(1.0))
#         ## mean_x
#         mean_x = self.box_filter(x_lr)/N
#         ## mean_y
#         mean_y = self.box_filter(y_lr)/N
#         ## cov_xy
#         cov_xy = self.box_filter(x_lr * y_lr)/N - mean_x * mean_y
#         ## var_x
#         var_x  = self.box_filter(x_lr * x_lr)/N - mean_x * mean_x
#
#         ## A
#         A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
#         ## b
#         b = mean_y - A * mean_x
#
#         ## mean_A; mean_b
#         mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
#         mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
#
#         return mean_A * x_hr + mean_b
