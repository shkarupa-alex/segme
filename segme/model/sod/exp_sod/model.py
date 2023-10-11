import numpy as np
from keras import backend, initializers, layers, models
from segme.common.align import Align
from segme.common.backbone import Backbone
from segme.common.convnormact import Conv, Norm, Act
from segme.common.head import HeadProjection, ClassificationActivation, ClassificationUncertainty
from segme.common.resize import BilinearInterpolation
from segme.common.fold import UnFold
from segme.common.split import Split
from segme.policy.backbone.diy.softswin import AttnBlock, MLP


# SwinBlock(
#         filters, current_window, pretrain_window, num_heads, shift_mode, qk_units=16, kernel_size=3, path_drop=0.,
#         expand_ratio=3., path_gamma=1., name=None)
# AttnBlock(current_window, pretrain_window, num_heads, shift_mode, path_drop=0., expand_ratio=4.,
#               path_gamma=1., name=None)
# MLPConv(filters, fused, kernel_size=3, expand_ratio=3., path_drop=0., gamma_initializer='ones', name=None)
# MLP    (expand_ratio=4., path_drop=0., gamma_initializer='ones', name=None)
def Attention(depth, window_size, shift_mode, expand_ratio=3., path_drop=0., path_gamma=1., name=None):
    if name is None:
        counter = backend.get_uid('attn')
        name = f'attn_{counter}'

    if isinstance(path_gamma, float):
        path_gamma = [path_gamma] * depth
    elif len(path_gamma) != depth:
        raise ValueError('Number of path gammas must equals to depth.')

    if isinstance(path_drop, float):
        path_drop = [path_drop] * depth
    elif len(path_drop) != depth:
        raise ValueError('Number of path dropouts must equals to depth.')

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        num_heads = channels // 32

        x = inputs
        for i in range(depth):
            if i % 2:
                current_shift = (shift_mode + i // 2 - 1) % 4 + 1
                x = AttnBlock(
                    window_size, window_size, num_heads, current_shift, path_drop=path_drop[i],
                    expand_ratio=expand_ratio, path_gamma=path_gamma[i], name=f'{name}_{i}')(x)
            else:
                x = AttnBlock(
                    window_size, window_size, num_heads, 0, path_drop=path_drop[i], expand_ratio=expand_ratio,
                    path_gamma=path_gamma[i], name=f'{name}_{i}')(x)

        return x

    return apply


def Convolution(depth, expand_ratio=2., path_drop=0., path_gamma=1., name=None):
    if name is None:
        counter = backend.get_uid('conv')
        name = f'conv_{counter}'

    if isinstance(path_gamma, float):
        path_gamma = [path_gamma] * depth
    elif len(path_gamma) != depth:
        raise ValueError('Number of path gammas must equals to depth.')

    if isinstance(path_drop, float):
        path_drop = [path_drop] * depth
    elif len(path_drop) != depth:
        raise ValueError('Number of path dropouts must equals to depth.')

    def apply(inputs):
        x = inputs
        for i in range(depth):
            gamma_initializer = initializers.Constant(path_gamma[i])
            x = MLP(
                expand_ratio=expand_ratio, path_drop=path_drop[i], gamma_initializer=gamma_initializer,
                name=f'{name}_conv_{i}')(x)

        return x

    return apply


def Head(unfold, depth, unknown, stride, kernel=1, name=None):
    if name is None:
        counter = backend.get_uid('head')
        name = f'head_{counter}'

    size = 1 + int(depth) + int(unknown)
    tasks = ['salient'] + ['depth'] * int(depth) + ['unknown'] * int(unknown)

    def apply(inputs):
        if unfold:
            x = HeadProjection(size * stride ** 2, kernel_size=kernel, name=f'{name}_logits')(inputs)
            x = UnFold(stride, name=f'{name}_unfold')(x)
        else:
            x = HeadProjection(size, kernel_size=kernel, name=f'{name}_logits')(inputs)
            x = BilinearInterpolation(stride, name=f'{name}_resize')(x)

        x = [x] if 1 == size else Split(size, name=f'{name}_split')(x)

        if unknown:
            u = ClassificationUncertainty(1, True, name=f'{name}_salient_unknown')(x[0])
            xu = layers.concatenate([u, x[-1]], name=f'{name}_unknown_concat')
            xu = HeadProjection(1, kernel_size=kernel, name=f'{name}_unknown_merge')(xu)
            x[-1] = xu

        x = [ClassificationActivation(name=f'{name}_{task}')(y) for y, task in zip(x, tasks)]

        return x

    return apply


def ExpSOD(
        sup_unfold=False, with_depth=False, with_unknown=False, transform_depth=2, window_size=24, path_gamma=0.01,
        path_drop=0.2):
    inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    outputs = Backbone()(inputs)[::-1]
    strides = [32, 16, 8, 4, 2, 1][:len(outputs)]

    num_shifts = transform_depth // 3 + transform_depth % 3 // 2
    path_size = transform_depth * 2 * (len(strides) - 1)
    path_gammas = np.linspace(1e-5, path_gamma, path_size).tolist()
    path_drops = np.linspace(path_drop, 0., path_size).tolist()

    heads, o_prev = [], None
    for i, (o, s) in enumerate(zip(outputs, strides)):
        channels = o.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        o = Conv(channels, 1, name=f'decoder_{s}_feature_proj')(o)
        o = Norm(name=f'decoder_{s}_feature_norm')(o)
        o = Act(name=f'decoder_{s}_feature_act')(o)

        if 32 == s:
            heads.extend(Head(sup_unfold, with_depth, with_unknown, s, name=f'head_{s}')(o))
            o_prev = o
            continue

        stage_drops, path_drops = path_drops[:transform_depth], path_drops[transform_depth:]
        stage_gammas, path_gammas = path_gammas[:transform_depth], path_gammas[transform_depth:]

        if 2 == s:
            o = Convolution(
                transform_depth, path_drop=stage_drops, path_gamma=stage_gammas,
                name=f'decoder_{s}_feature_transform')(o)
        else:
            shift_mode = num_shifts * i * 2 % 4 + 1
            o = Attention(
                transform_depth, window_size, shift_mode, path_drop=stage_drops, path_gamma=stage_gammas,
                name=f'decoder_{s}_feature_transform')(o)

        o = Align(channels, name=f'decoder_{s}_merge')([o, o_prev])

        stage_drops, path_drops = path_drops[:transform_depth], path_drops[transform_depth:]
        stage_gammas, path_gammas = path_gammas[:transform_depth], path_gammas[transform_depth:]

        if 2 == s:
            o = Convolution(
                transform_depth, path_drop=stage_drops, path_gamma=stage_gammas,
                name=f'decoder_{s}_combo_transform')(o)
        else:
            shift_mode = (num_shifts * i * 2 + num_shifts) % 4 + 1
            o = Attention(
                transform_depth, window_size, shift_mode, path_drop=stage_drops, path_gamma=stage_gammas,
                name=f'decoder_{s}_combo_transform')(o)

        heads.extend(Head(sup_unfold, with_depth, with_unknown, s, name=f'head_{s}')(o))
        o_prev = o

    model = models.Model(inputs=inputs, outputs=heads, name='exp_sod')

    return model
