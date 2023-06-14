from keras import backend, layers, models
from segme.common.adppool import AdaptiveAveragePooling
from segme.common.align import Align
from segme.common.backbone import Backbone
from segme.common.convnormact import ConvNormAct
from segme.common.head import HeadProjection, ClassificationActivation
from segme.common.resize import NearestInterpolation, BilinearInterpolation
from segme.common.sequence import Sequence
from segme.common.unfold import UnFold
from segme.policy.backbone.utils import patch_config


def Encoder():
    base_model = Backbone([2, 4, 32], 3, 'resnet_rs_50_s8-imagenet')
    base_config = base_model.get_config()
    base_weights = base_model.get_weights()

    ext_config = patch_config(base_config, [0], 'batch_input_shape', lambda old: old[:-1] + (4,))
    ext_config = patch_config(ext_config, [2], 'mean', lambda old: old + [0.449])
    ext_config = patch_config(ext_config, [2], 'variance', lambda old: old + [0.497 ** 2])
    ext_model = models.Model.from_config(ext_config)

    ext_weights = []
    for base_weight, ext_weight in zip(base_weights, ext_model.get_weights()):
        if base_weight.shape != ext_weight.shape:
            if base_weight.shape[:2] + base_weight.shape[3:] != ext_weight.shape[:2] + ext_weight.shape[3:]:
                raise ValueError('Unexpected weight shape')

            ext_weight[:, :, :base_weight.shape[2]] = base_weight
            ext_weights.append(ext_weight)
        else:
            ext_weights.append(base_weight)

    ext_model.set_weights(ext_weights)
    ext_model.trainable = True

    return ext_model


def FPP(kernel_size=3, name=None):
    if name is None:
        counter = backend.get_uid('lmpp')
        name = f'lmpp_{counter}'

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        x = [ConvNormAct(channels, kernel_size, name=f'{name}_cna')(inputs)]

        atr_channels = (channels // 24) * 8
        for atr_rate in [12, 24, 36]:
            y = Sequence([
                ConvNormAct(None, kernel_size, dilation_rate=atr_rate, name=f'{name}_atr{atr_rate}_dna'),
                ConvNormAct(atr_channels, 1, name=f'{name}_atr{atr_rate}_pna'),
            ], name=f'{name}_atr{atr_rate}')(inputs)
            x.append(y)

        avg_channels = (channels // 32) * 8
        for avg_rate in [1, 2, 3, 6]:
            y = Sequence([
                AdaptiveAveragePooling(avg_rate, name=f'{name}_avg{avg_rate}_pool'),
                ConvNormAct(avg_channels, 1, name=f'{name}_avg{avg_rate}_pna')
            ], name=f'{name}_avg{avg_rate}')(inputs)
            if 1 == avg_rate:
                y = NearestInterpolation()([y, inputs])
            else:
                y = BilinearInterpolation()([y, inputs])
            x.append(y)

        x = layers.concatenate(x, name=f'{name}_merge')
        x = ConvNormAct(channels, 1, name=f'{name}_proj')(x)

        return x

    return apply


def Head(unfold, stride, name=None):
    if name is None:
        counter = backend.get_uid('head')
        name = f'head_{counter}'

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        if unfold:
            x = HeadProjection(stride ** 2, name=f'{name}_logits')(inputs)
            x = UnFold(stride, name=f'{name}_unfold')(x)
        else:
            x = HeadProjection(1, name=f'{name}_logits')(inputs)
            x = BilinearInterpolation(stride, name=f'{name}_resize')(x)

        x = ClassificationActivation(name=f'{name}_act')(x)

        return x

    return apply


def ExpRef(sup_unfold=False):
    inputs = layers.Input(name='image', shape=[None, None, 4], dtype='uint8')
    feats2, feats4, feats8 = Encoder()(inputs)

    outputs8 = FPP(name='fpp')(feats8)
    probs8 = Head(sup_unfold, 8, name='head8')(outputs8)

    outputs4 = Align(feats4.shape[-1], name='merge4')([feats4, outputs8])
    probs4 = Head(sup_unfold, 4, name='head4')(outputs4)

    outputs2 = Align(feats2.shape[-1], name='merge2')([feats2, outputs4])
    probs2 = Head(sup_unfold, 2, name='head2')(outputs2)

    model = models.Model(inputs=inputs, outputs=[probs2, probs4, probs8], name='exp_sod')

    return model
