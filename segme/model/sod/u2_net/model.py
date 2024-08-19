import numpy as np
from keras.src import layers
from keras.src import models
from keras.src.utils import naming

from segme.common.convnormact import ConvNormAct
from segme.common.head import ClassificationActivation
from segme.common.head import ClassificationHead
from segme.common.head import HeadProjection
from segme.common.resize import BilinearInterpolation
from segme.policy import dtpol


def _RSU7(mid_features, out_features, name=None):
    if name is None:
        counter = naming.get_uid("rsu7")
        name = f"rsu7_{counter}"

    def apply(x):
        x0 = ConvNormAct(out_features, 3, name=f"{name}_cna0")(x)

        x1 = ConvNormAct(mid_features, 3, name=f"{name}_cna1")(x0)

        x2 = layers.MaxPooling2D(2, padding="same", name=f"{name}_pool1")(x1)
        x2 = ConvNormAct(mid_features, 3, name=f"{name}_cna2")(x2)

        x3 = layers.MaxPooling2D(2, padding="same", name=f"{name}_pool2")(x2)
        x3 = ConvNormAct(mid_features, 3, name=f"{name}_cna3")(x3)

        x4 = layers.MaxPooling2D(2, padding="same", name=f"{name}_pool3")(x3)
        x4 = ConvNormAct(mid_features, 3, name=f"{name}_cna4")(x4)

        x5 = layers.MaxPooling2D(2, padding="same", name=f"{name}_pool4")(x4)
        x5 = ConvNormAct(mid_features, 3, name=f"{name}_cna5")(x5)

        x6 = layers.MaxPooling2D(2, padding="same", name=f"{name}_pool5")(x5)
        x6 = ConvNormAct(mid_features, 3, name=f"{name}_cna6")(x6)

        x7 = ConvNormAct(mid_features, 3, dilation_rate=2, name=f"{name}_cna7")(
            x6
        )

        x6d = layers.concatenate([x7, x6], axis=-1, name=f"{name}_concat6d")
        x6d = ConvNormAct(mid_features, 3, name=f"{name}_cna6d")(x6d)

        x5d = BilinearInterpolation(2, name=f"{name}_resize5d")(x6d)
        x5d = layers.concatenate([x5d, x5], axis=-1, name=f"{name}_concat5d")
        x5d = ConvNormAct(mid_features, 3, name=f"{name}_cna5d")(x5d)

        x4d = BilinearInterpolation(2, name=f"{name}_resize4d")(x5d)
        x4d = layers.concatenate([x4d, x4], axis=-1, name=f"{name}_concat4d")
        x4d = ConvNormAct(mid_features, 3, name=f"{name}_cna4d")(x4d)

        x3d = BilinearInterpolation(2, name=f"{name}_resize3d")(x4d)
        x3d = layers.concatenate([x3d, x3], axis=-1, name=f"{name}_concat3d")
        x3d = ConvNormAct(mid_features, 3, name=f"{name}_cna3d")(x3d)

        x2d = BilinearInterpolation(2, name=f"{name}_resize2d")(x3d)
        x2d = layers.concatenate([x2d, x2], axis=-1, name=f"{name}_concat2d")
        x2d = ConvNormAct(mid_features, 3, name=f"{name}_cna2d")(x2d)

        x1d = BilinearInterpolation(2, name=f"{name}_resize1d")(x2d)
        x1d = layers.concatenate([x1d, x1], axis=-1, name=f"{name}_concat1d")
        x1d = ConvNormAct(out_features, 3, name=f"{name}_cna1d")(x1d)

        return layers.add([x1d, x0], name=f"{name}_add")

    return apply


def _RSU6(mid_features, out_features, name=None):
    if name is None:
        counter = naming.get_uid("rsu6")
        name = f"rsu6_{counter}"

    def apply(x):
        x0 = ConvNormAct(out_features, 3, name=f"{name}_cna0")(x)

        x1 = ConvNormAct(mid_features, 3, name=f"{name}_cna1")(x0)

        x2 = layers.MaxPooling2D(2, padding="same", name=f"{name}_pool1")(x1)
        x2 = ConvNormAct(mid_features, 3, name=f"{name}_cna2")(x2)

        x3 = layers.MaxPooling2D(2, padding="same", name=f"{name}_pool2")(x2)
        x3 = ConvNormAct(mid_features, 3, name=f"{name}_cna3")(x3)

        x4 = layers.MaxPooling2D(2, padding="same", name=f"{name}_pool3")(x3)
        x4 = ConvNormAct(mid_features, 3, name=f"{name}_cna4")(x4)

        x5 = layers.MaxPooling2D(2, padding="same", name=f"{name}_pool4")(x4)
        x5 = ConvNormAct(mid_features, 3, name=f"{name}_cna5")(x5)

        x6 = ConvNormAct(mid_features, 3, dilation_rate=2, name=f"{name}_cna6")(
            x5
        )

        x5d = layers.concatenate([x6, x5], axis=-1, name=f"{name}_concat5d")
        x5d = ConvNormAct(mid_features, 3, name=f"{name}_cna5d")(x5d)

        x4d = BilinearInterpolation(2, name=f"{name}_resize4d")(x5d)
        x4d = layers.concatenate([x4d, x4], axis=-1, name=f"{name}_concat4d")
        x4d = ConvNormAct(mid_features, 3, name=f"{name}_cna4d")(x4d)

        x3d = BilinearInterpolation(2, name=f"{name}_resize3d")(x4d)
        x3d = layers.concatenate([x3d, x3], axis=-1, name=f"{name}_concat3d")
        x3d = ConvNormAct(mid_features, 3, name=f"{name}_cna3d")(x3d)

        x2d = BilinearInterpolation(2, name=f"{name}_resize2d")(x3d)
        x2d = layers.concatenate([x2d, x2], axis=-1, name=f"{name}_concat2d")
        x2d = ConvNormAct(mid_features, 3, name=f"{name}_cna2d")(x2d)

        x1d = BilinearInterpolation(2, name=f"{name}_resize1d")(x2d)
        x1d = layers.concatenate([x1d, x1], axis=-1, name=f"{name}_concat1d")
        x1d = ConvNormAct(out_features, 3, name=f"{name}_cna1d")(x1d)

        return layers.add([x1d, x0])

    return apply


def _RSU5(mid_features, out_features, name=None):
    if name is None:
        counter = naming.get_uid("rsu5")
        name = f"rsu5_{counter}"

    def apply(x):
        x0 = ConvNormAct(out_features, 3, name=f"{name}_cna0")(x)

        x1 = ConvNormAct(mid_features, 3, name=f"{name}_cna1")(x0)

        x2 = layers.MaxPooling2D(2, padding="same", name=f"{name}_pool1")(x1)
        x2 = ConvNormAct(mid_features, 3, name=f"{name}_cna2")(x2)

        x3 = layers.MaxPooling2D(2, padding="same", name=f"{name}_pool2")(x2)
        x3 = ConvNormAct(mid_features, 3, name=f"{name}_cna3")(x3)

        x4 = layers.MaxPooling2D(2, padding="same", name=f"{name}_pool3")(x3)
        x4 = ConvNormAct(mid_features, 3, name=f"{name}_cna4")(x4)

        x5 = ConvNormAct(mid_features, 3, dilation_rate=2, name=f"{name}_cna5")(
            x4
        )

        x4d = layers.concatenate([x5, x4], axis=-1, name=f"{name}_concat4d")
        x4d = ConvNormAct(mid_features, 3, name=f"{name}_cna4d")(x4d)

        x3d = BilinearInterpolation(2, name=f"{name}_resize3d")(x4d)
        x3d = layers.concatenate([x3d, x3], axis=-1, name=f"{name}_concat3d")
        x3d = ConvNormAct(mid_features, 3, name=f"{name}_cna3d")(x3d)

        x2d = BilinearInterpolation(2, name=f"{name}_resize2d")(x3d)
        x2d = layers.concatenate([x2d, x2], axis=-1, name=f"{name}_concat2d")
        x2d = ConvNormAct(mid_features, 3, name=f"{name}_cna2d")(x2d)

        x1d = BilinearInterpolation(2, name=f"{name}_resize1d")(x2d)
        x1d = layers.concatenate([x1d, x1], axis=-1, name=f"{name}_concat1d")
        x1d = ConvNormAct(out_features, 3, name=f"{name}_cna1d")(x1d)

        return layers.add([x1d, x0], name=f"{name}_add")

    return apply


def _RSU4(mid_features, out_features, name=None):
    if name is None:
        counter = naming.get_uid("rsu4")
        name = f"rsu4_{counter}"

    def apply(x):
        x0 = ConvNormAct(out_features, 3, name=f"{name}_cna0")(x)

        x1 = ConvNormAct(mid_features, 3, name=f"{name}_cna1")(x0)

        x2 = layers.MaxPooling2D(2, padding="same", name=f"{name}_pool1")(x1)
        x2 = ConvNormAct(mid_features, 3, name=f"{name}_cna2")(x2)

        x3 = layers.MaxPooling2D(2, padding="same", name=f"{name}_pool2")(x2)
        x3 = ConvNormAct(mid_features, 3, name=f"{name}_cna3")(x3)

        x4 = ConvNormAct(mid_features, 3, dilation_rate=2, name=f"{name}_cna4")(
            x3
        )

        x3d = layers.concatenate([x4, x3], axis=-1, name=f"{name}_concat3d")
        x3d = ConvNormAct(mid_features, 3, name=f"{name}_cna3d")(x3d)

        x2d = BilinearInterpolation(2, name=f"{name}_resize2d")(x3d)
        x2d = layers.concatenate([x2d, x2], axis=-1, name=f"{name}_concat2d")
        x2d = ConvNormAct(mid_features, 3, name=f"{name}_cna2d")(x2d)

        x1d = BilinearInterpolation(2, name=f"{name}_resize1d")(x2d)
        x1d = layers.concatenate([x1d, x1], axis=-1, name=f"{name}_concat1d")
        x1d = ConvNormAct(out_features, 3, name=f"{name}_cna1d")(x1d)

        return layers.add([x1d, x0], name=f"{name}_add")

    return apply


def _RSU4f(mid_features, out_features, name=None):
    if name is None:
        counter = naming.get_uid("rsu4f")
        name = f"rsu4f_{counter}"

    def apply(x):
        x0 = ConvNormAct(out_features, 3, name=f"{name}_cna0")(x)

        x1 = ConvNormAct(mid_features, 3, name=f"{name}_cna1")(x0)

        x2 = ConvNormAct(mid_features, 3, dilation_rate=2, name=f"{name}_cna2")(
            x1
        )

        x3 = ConvNormAct(mid_features, 3, dilation_rate=4, name=f"{name}_cna3")(
            x2
        )

        x4 = ConvNormAct(mid_features, 3, dilation_rate=8, name=f"{name}_cna4")(
            x3
        )

        x3d = layers.concatenate([x4, x3], axis=-1, name=f"{name}_concat3d")
        x3d = ConvNormAct(
            mid_features, 3, dilation_rate=4, name=f"{name}_cna3d"
        )(x3d)

        x2d = layers.concatenate([x3d, x2], axis=-1, name=f"{name}_concat2d")
        x2d = ConvNormAct(
            mid_features, 3, dilation_rate=2, name=f"{name}_cna2d"
        )(x2d)

        x1d = layers.concatenate([x2d, x1], axis=-1, name=f"{name}_concat1d")
        x1d = ConvNormAct(out_features, 3, name=f"{name}_cna1d")(x1d)

        return layers.add([x1d, x0], name=f"{name}_add")

    return apply


def U2Net(classes, dtype=None):
    if dtype is not None:
        with dtpol.policy_scope(dtype):
            return U2Net(classes, dtype=None)

    inputs = layers.Input(name="image", shape=(None, None, 3), dtype="uint8")

    x = layers.Normalization(
        mean=np.array([0.485, 0.456, 0.406], "float32") * 255.0,
        variance=(np.array([0.229, 0.224, 0.225], "float32") * 255.0) ** 2,
        name="normalize",
    )(inputs)

    x1 = _RSU7(32, 64, name="stage1")(x)

    x2 = layers.MaxPooling2D(2, padding="same", name="pool1")(x1)
    x2 = _RSU6(32, 128, name="stage2")(x2)

    x3 = layers.MaxPooling2D(2, padding="same", name="pool2")(x2)
    x3 = _RSU5(64, 256, name="stage3")(x3)

    x4 = layers.MaxPooling2D(2, padding="same", name="pool3")(x3)
    x4 = _RSU4(128, 512, name="stage4")(x4)

    x5 = layers.MaxPooling2D(2, padding="same", name="pool4")(x4)
    x5 = _RSU4f(256, 512, name="stage5")(x5)

    x6 = layers.MaxPooling2D(2, padding="same", name="pool5")(x5)
    x6 = _RSU4f(256, 512, name="stage6")(x6)

    x5d = BilinearInterpolation(2, name="resize5d")(x6)
    x5d = layers.concatenate([x5d, x5], axis=-1, name="concat5d")
    x5d = _RSU4f(256, 512, name="stage5d")(x5d)

    x4d = BilinearInterpolation(2, name="resize4d")(x5d)
    x4d = layers.concatenate([x4d, x4], axis=-1, name="concat4d")
    x4d = _RSU4(128, 256, name="stage4d")(x4d)

    x3d = BilinearInterpolation(2, name="resize3d")(x4d)
    x3d = layers.concatenate([x3d, x3], axis=-1, name="concat3d")
    x3d = _RSU5(64, 128, name="stage3d")(x3d)

    x2d = BilinearInterpolation(2, name="resize2d")(x3d)
    x2d = layers.concatenate([x2d, x2], axis=-1, name="concat2d")
    x2d = _RSU6(32, 64, name="stage2d")(x2d)

    x1d = BilinearInterpolation(2, name="resize1d")(x2d)
    x1d = layers.concatenate([x1d, x1], axis=-1, name="concat1d")
    x1d = _RSU7(16, 64, name="stage1d")(x1d)

    heads = [x1d, x2d, x3d, x4d, x5d, x6]
    heads = [
        HeadProjection(classes, 3, name=f"stage{i + 1}d_proj")(h)
        for i, h in enumerate(heads)
    ]
    heads = heads[:1] + [
        BilinearInterpolation(2 ** (i + 1), name=f"stage{i + 2}d_resize")(h)
        for i, h in enumerate(heads[1:])
    ]

    head0 = layers.concatenate(heads, axis=-1, name="stage0d_concat")
    head0 = ClassificationHead(classes, name="stage0d_head_act")(head0)

    heads = [
        ClassificationActivation(name=f"stage{i + 1}d_act")(h)
        for i, h in enumerate(heads)
    ]

    outputs = (head0,) + tuple(heads)

    model = models.Model(inputs=inputs, outputs=outputs, name="u2net")

    return model


def U2NetP(classes, dtype=None):
    if dtype is not None:
        with dtpol.policy_scope(dtype):
            return U2NetP(classes, dtype=None)

    inputs = layers.Input(name="image", shape=(None, None, 3), dtype="uint8")

    x = layers.Normalization(
        mean=np.array([0.485, 0.456, 0.406], "float32") * 255.0,
        variance=(np.array([0.229, 0.224, 0.225], "float32") * 255.0) ** 2,
        name="normalize",
    )(inputs)

    x1 = _RSU7(16, 64, name="stage1")(x)

    x2 = layers.MaxPooling2D(2, padding="same", name="pool1")(x1)
    x2 = _RSU6(16, 64, name="stage2")(x2)

    x3 = layers.MaxPooling2D(2, padding="same", name="pool2")(x2)
    x3 = _RSU5(16, 64, name="stage3")(x3)

    x4 = layers.MaxPooling2D(2, padding="same", name="pool3")(x3)
    x4 = _RSU4(16, 64, name="stage4")(x4)

    x5 = layers.MaxPooling2D(2, padding="same", name="pool4")(x4)
    x5 = _RSU4f(16, 64, name="stage5")(x5)

    x6 = layers.MaxPooling2D(2, padding="same", name="pool5")(x5)
    x6 = _RSU4f(16, 64, name="stage6")(x6)

    x5d = BilinearInterpolation(2, name="resize5d")(x6)
    x5d = layers.concatenate([x5d, x5], axis=-1, name="concat5d")
    x5d = _RSU4f(16, 64, name="stage5d")(x5d)

    x4d = BilinearInterpolation(2, name="resize4d")(x5d)
    x4d = layers.concatenate([x4d, x4], axis=-1, name="concat4d")
    x4d = _RSU4(16, 64, name="stage4d")(x4d)

    x3d = BilinearInterpolation(2, name="resize3d")(x4d)
    x3d = layers.concatenate([x3d, x3], axis=-1, name="concat3d")
    x3d = _RSU5(16, 64, name="stage3d")(x3d)

    x2d = BilinearInterpolation(2, name="resize2d")(x3d)
    x2d = layers.concatenate([x2d, x2], axis=-1, name="concat2d")
    x2d = _RSU6(16, 64, name="stage2d")(x2d)

    x1d = BilinearInterpolation(2, name="resize1d")(x2d)
    x1d = layers.concatenate([x1d, x1], axis=-1, name="concat1d")
    x1d = _RSU7(16, 64, name="stage1d")(x1d)

    heads = [x1d, x2d, x3d, x4d, x5d, x6]
    heads = [
        HeadProjection(classes, 3, name=f"stage{i + 1}d_proj")(h)
        for i, h in enumerate(heads)
    ]
    heads = heads[:1] + [
        BilinearInterpolation(2 ** (i + 1), name=f"stage{i + 2}d_resize")(h)
        for i, h in enumerate(heads[1:])
    ]

    head0 = layers.concatenate(heads, axis=-1, name="stage0d_concat")
    head0 = ClassificationHead(classes, name="stage0d_head_act")(head0)

    heads = [
        ClassificationActivation(name=f"stage{i + 1}d_act")(h)
        for i, h in enumerate(heads)
    ]

    outputs = (head0,) + tuple(heads)

    model = models.Model(inputs=inputs, outputs=outputs, name="u2netp")

    return model
