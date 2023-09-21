from itertools import cycle
from keras import backend, layers, models
from keras.saving import register_keras_serializable
from segme.common.align import Align
from segme.common.attn import SwinAttention
from segme.common.backbone import Backbone
from segme.common.convnormact import Conv, Norm, Act
from segme.common.head import HeadProjection, ClassificationActivation, ClassificationUncertainty
from segme.common.resize import BilinearInterpolation
from segme.common.fold import Fold, UnFold


def Transformer(window_size, shift_mode, expand_ratio=4., name=None):
    if name is None:
        counter = backend.get_uid('transformer')
        name = f'transformer_{counter}'

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        num_heads = channels // 32
        x = SwinAttention(window_size, window_size, num_heads, shift_mode, name=f'{name}_swin_attn')(inputs)
        x = Norm(name=f'{name}_swin_norm')(x)
        x = layers.add([x, inputs], name=f'{name}_swin_add')

        expand_filters = int(channels * expand_ratio)
        y = layers.Dense(expand_filters, name=f'{name}_expand')(x)
        y = Act(name=f'{name}_act')(y)
        y = layers.Dense(channels, name=f'{name}_squeeze')(y)
        y = Norm(name=f'{name}_norm')(y)
        y = layers.add([y, x], name=f'{name}_add')

        return y

    return apply


def FMBConv(expand_ratio=4., name=None):
    if name is None:
        counter = backend.get_uid('fmbconv')
        name = f'fmbconv_{counter}'

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        expand_filters = int(channels * expand_ratio)
        x = Conv(expand_filters, 3, name=f'{name}_expand')(inputs)
        x = Act(name=f'{name}_act')(x)
        x = Conv(channels, 1, name=f'{name}_squeeze')(x)
        x = Norm(name=f'{name}_norm')(x)
        x = layers.add([inputs, x], name=f'{name}_add')

        return x

    return apply


@register_keras_serializable(package='SegMe>Model>SOD>ExpSelRref')
class PlusOne(layers.Layer):
    def call(self, inputs, **kwargs):
        return inputs + 1.

    def compute_output_shape(self, input_shape):
        return input_shape


def CRM(fused, stride, sup_unfold, window_size, shift_dir, name=None):
    if name is None:
        counter = backend.get_uid('crm')
        name = f'crm_{counter}'

    def apply(inputs, head):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        s = Fold(stride, name=f'{name}_supervision_fold')(head)
        s = Conv(channels, 3, name=f'{name}_supervision_proj')(s)

        x = layers.concatenate([inputs, s], name=f'{name}_concat')
        for i in range(2):
            if fused:
                x = Conv(channels, 3, name=f'{name}_fuse_{i}')(x)
            else:
                x = Conv(channels, 1, name=f'{name}_fuse_pw_{i}')(x)
                x = Conv(channels, 3, name=f'{name}_fuse_dw_{i}')(x)
            x = Norm(name=f'{name}_fuse_norm_{i}')(x)
            x = Act(name=f'{name}_fuse_act_{i}')(x)

        h1 = Head(sup_unfold, stride, kernel=3, name=f'{name}_head1')(x)

        u = ClassificationUncertainty(2, False, name=f'{name}_uncertainty')(h1)
        u = Fold(stride, name=f'{name}_uncertainty_fold')(u)
        u = Conv(channels, 3, name=f'{name}_uncertainty_proj')(u)
        u = Norm(name=f'{name}_uncertainty_norm')(u)
        u = Act(name=f'{name}_uncertainty_act')(u)
        u = PlusOne(name=f'{name}_uncertainty_plus')(u)

        x = layers.multiply([x, u], name=f'{name}_highlight')

        x = Transformer(window_size, 0, name=f'{name}_post_transformer_0')(x)
        x = Transformer(window_size, shift_dir, name=f'{name}_post_transformer_1')(x)

        h2 = Head(sup_unfold, stride, kernel=3, name=f'{name}_head2')(x)

        return x, h1, h2

    return apply


def Head(unfold, stride, kernel=1, name=None):
    if name is None:
        counter = backend.get_uid('head')
        name = f'head_{counter}'

    def apply(inputs):
        if unfold:
            x = HeadProjection(stride ** 2, kernel_size=kernel, name=f'{name}_logits')(inputs)
            x = UnFold(stride, name=f'{name}_unfold')(x)
        else:
            x = HeadProjection(1, kernel_size=kernel, name=f'{name}_logits')(inputs)
            x = BilinearInterpolation(stride, name=f'{name}_resize')(x)

        x = ClassificationActivation(name=f'{name}_salient')(x)

        return x

    return apply


def ExpSelfRef(sup_unfold=False, window_size=24):
    # TODO: with_unknown=False, with_depth=False
    inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    features = Backbone()(inputs)[::-1]
    stages = [32, 16, 8, 4, 2, 1][:len(features)]

    outputs = []
    for f, s in zip(features, stages):
        channels = f.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        f = Norm(name=f'decoder_{s}_feature_norm')(f)
        f = Conv(channels, 1, name=f'decoder_{s}_feature_proj')(f)
        f = Act(name=f'decoder_{s}_feature_act')(f)

        outputs.append(f)

    output32, outputs, stages = outputs[0], outputs[1:], stages[1:]
    head32 = Head(sup_unfold, 32, name='head32')(output32)

    heads = []
    f_prev, h_prev = output32, head32
    for f, s, m in zip(outputs, stages, cycle(range(1, 5))):
        channels = f.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        if 2 == s:
            for i in range(3):
                f = FMBConv(name=f'decoder_{s}_fmbconv_{i}')(f)
        else:
            f = Transformer(window_size, 0, name=f'decoder_{s}_transformer_0')(f)
            f = Transformer(window_size, m, name=f'decoder_{s}_transformer_1')(f)

        f = Align(channels, name=f'decoder_{s}_merge')([f, f_prev])

        if 2 == s:
            f = FMBConv(name=f'decoder_{s}_fuse')(f)
        else:
            f = Transformer(window_size, 0, name=f'decoder_{s}_fuse')(f)

        f_prev, h_prev_, h_prev = CRM(
            s <= 4, s, sup_unfold, window_size, (s + 1) % 4 + 1, name=f'decoder_{s}_crm')(f, h_prev)

        heads.append(h_prev_)
        heads.append(h_prev)

    head2 = Head(sup_unfold, stages[-1], name='head2')(f_prev)

    outputs = [head32] + heads + [head2]

    model = models.Model(inputs=inputs, outputs=outputs[::-1], name='exp_self_ref')

    return model
