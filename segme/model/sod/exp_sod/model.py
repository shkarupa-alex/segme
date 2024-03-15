import numpy as np
from tf_keras import backend, layers, models
from segme.common.align import Align
from segme.common.backbone import Backbone
from segme.common.convnormact import Conv, Norm, Act
from segme.common.head import HeadProjection, ClassificationActivation, ClassificationUncertainty
from segme.common.resize import BilinearInterpolation
from segme.common.fold import Fold, UnFold
from segme.common.split import Split
from segme.policy import cnapol
from segme.policy.backbone.diy.hardswin import AttnBlock


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
            current_shift = (shift_mode + i // 2 - 1) % 4 + 1 if i % 2 else 0
            x = AttnBlock(
                window_size, window_size, num_heads, current_shift, path_drop=path_drop[i], expand_ratio=expand_ratio,
                path_gamma=path_gamma[i], name=f'{name}_{i}')(x)

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
        sup_unfold=False, with_depth=False, with_unknown=False, transform_depth=2, window_size=24, path_gamma=0.1,
        path_drop=0.2):
    backbone = Backbone()
    outputs = backbone.outputs[::-1]

    num_shifts = transform_depth // 3 + transform_depth % 3 // 2
    path_size = transform_depth * 2 * (len(outputs) - 1)
    path_gammas = np.linspace(1e-5, path_gamma, path_size).tolist()
    path_drops = np.linspace(path_drop, 0., path_size).tolist()

    with cnapol.policy_scope('conv-ln1em5-gelu'):
        heads, o_prev = [], None
        for i, o in enumerate(outputs):
            channels = o.shape[-1]
            if channels is None:
                raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

            stride = 2 ** (5 - i)

            o = Conv(channels, 1, name=f'backstage_{i}_lateral_proj')(o)
            o = Act(name=f'backstage_{i}_lateral_act')(o)
            o = Norm(name=f'backstage_{i}_lateral_norm')(o)

            if o_prev is None:
                o_prev = o
                heads.extend(Head(sup_unfold, with_depth, with_unknown, stride, name=f'head_{i}')(o))
                continue

            shift_mode = num_shifts * (i - 1) * 2 % 4 + 1
            stage_drops, path_drops = path_drops[:transform_depth], path_drops[transform_depth:]
            stage_gammas, path_gammas = path_gammas[:transform_depth], path_gammas[transform_depth:]
            o = Attention(
                transform_depth, window_size, shift_mode, path_drop=stage_drops, path_gamma=stage_gammas,
                name=f'backstage_{i}_lateral_transform')(o)

            o = Align(channels, name=f'backstage_{i}_merge_align')([o, o_prev])
            o = Norm(name=f'backstage_{i}_merge_norm')(o)

            shift_mode = (num_shifts * (i - 1) * 2 + num_shifts) % 4 + 1
            stage_drops, path_drops = path_drops[:transform_depth], path_drops[transform_depth:]
            stage_gammas, path_gammas = path_gammas[:transform_depth], path_gammas[transform_depth:]
            o = Attention(
                transform_depth, window_size, shift_mode, path_drop=stage_drops, path_gamma=stage_gammas,
                name=f'backstage_{i}_merge_transform')(o)

            # TODO: last head proj with kernel=5?
            o_prev = o
            heads.extend(Head(sup_unfold, with_depth, with_unknown, stride, name=f'head_{i}')(o))

        model = models.Model(inputs=backbone.inputs, outputs=heads, name='exp_sod')

        return model
