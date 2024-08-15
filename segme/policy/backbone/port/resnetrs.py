from functools import partial

import sys
import tensorflow as tf
from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src.applications import imagenet_utils

from keras.src.dtype_policies import dtype_policy
from keras.src.ops import operation_utils
from keras.src.utils import file_utils

from segme.common.convnormact import Act
from segme.common.convnormact import Conv
from segme.common.convnormact import Norm
from segme.policy import cnapol
from segme.policy.backbone.backbone import BACKBONES
from segme.policy.backbone.utils import patch_config
from segme.policy.backbone.utils import wrap_bone

BASE_WEIGHTS_URL = (
    "https://storage.googleapis.com/tensorflow/keras-applications/resnet_rs/"
)

WEIGHT_HASHES = {
    "resnet-rs-101-i160.h5": "544b3434d00efc199d66e9058c7f3379",
    "resnet-rs-101-i160_notop.h5": "82d5b90c5ce9d710da639d6216d0f979",
    "resnet-rs-101-i192.h5": "eb285be29ab42cf4835ff20a5e3b5d23",
    "resnet-rs-101-i192_notop.h5": "f9a0f6b85faa9c3db2b6e233c4eebb5b",
    "resnet-rs-152-i192.h5": "8d72a301ed8a6f11a47c4ced4396e338",
    "resnet-rs-152-i192_notop.h5": "5fbf7ac2155cb4d5a6180ee9e3aa8704",
    "resnet-rs-152-i224.h5": "31a46a92ab21b84193d0d71dd8c3d03b",
    "resnet-rs-152-i224_notop.h5": "dc8b2cba2005552eafa3167f00dc2133",
    "resnet-rs-152-i256.h5": "ba6271b99bdeb4e7a9b15c05964ef4ad",
    "resnet-rs-152-i256_notop.h5": "fa79794252dbe47c89130f65349d654a",
    "resnet-rs-200-i256.h5": "a76930b741884e09ce90fa7450747d5f",
    "resnet-rs-200-i256_notop.h5": "bbdb3994718dfc0d1cd45d7eff3f3d9c",
    "resnet-rs-270-i256.h5": "20d575825ba26176b03cb51012a367a8",
    "resnet-rs-270-i256_notop.h5": "2c42ecb22e35f3e23d2f70babce0a2aa",
    "resnet-rs-350-i256.h5": "f4a039dc3c421321b7fc240494574a68",
    "resnet-rs-350-i256_notop.h5": "6e44b55025bbdff8f51692a023143d66",
    "resnet-rs-350-i320.h5": "7ccb858cc738305e8ceb3c0140bee393",
    "resnet-rs-350-i320_notop.h5": "ab0c1f9079d2f85a9facbd2c88aa6079",
    "resnet-rs-420-i320.h5": "ae0eb9bed39e64fc8d7e0db4018dc7e8",
    "resnet-rs-420-i320_notop.h5": "fe6217c32be8305b1889657172b98884",
    "resnet-rs-50-i160.h5": "69d9d925319f00a8bdd4af23c04e4102",
    "resnet-rs-50-i160_notop.h5": "90daa68cd26c95aa6c5d25451e095529",
}

DEPTH_TO_WEIGHT_VARIANTS = {
    50: [160],
    101: [160, 192],
    152: [192, 224, 256],
    200: [256],
    270: [256],
    350: [256, 320],
    420: [320],
}
BLOCK_ARGS = {
    50: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 4},
        {"input_filters": 256, "num_repeats": 6},
        {"input_filters": 512, "num_repeats": 3},
    ],
    101: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 4},
        {"input_filters": 256, "num_repeats": 23},
        {"input_filters": 512, "num_repeats": 3},
    ],
    152: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 8},
        {"input_filters": 256, "num_repeats": 36},
        {"input_filters": 512, "num_repeats": 3},
    ],
    200: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 24},
        {"input_filters": 256, "num_repeats": 36},
        {"input_filters": 512, "num_repeats": 3},
    ],
    270: [
        {"input_filters": 64, "num_repeats": 4},
        {"input_filters": 128, "num_repeats": 29},
        {"input_filters": 256, "num_repeats": 53},
        {"input_filters": 512, "num_repeats": 4},
    ],
    350: [
        {"input_filters": 64, "num_repeats": 4},
        {"input_filters": 128, "num_repeats": 36},
        {"input_filters": 256, "num_repeats": 72},
        {"input_filters": 512, "num_repeats": 4},
    ],
    420: [
        {"input_filters": 64, "num_repeats": 4},
        {"input_filters": 128, "num_repeats": 44},
        {"input_filters": 256, "num_repeats": 87},
        {"input_filters": 512, "num_repeats": 4},
    ],
}

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal",
    },
}

CONV_KWARGS = {"use_bias": False, "kernel_initializer": CONV_KERNEL_INITIALIZER}

def get_survival_probability(init_rate, block_num, total_blocks):
    """Get survival probability based on block number and initial rate."""
    return init_rate * float(block_num) / total_blocks


def allow_bigger_recursion(target_limit: int):
    """Increase default recursion limit to create larger models."""
    current_limit = sys.getrecursionlimit()
    if current_limit < target_limit:
        sys.setrecursionlimit(target_limit)


def norm_kwargs():
    if "bn" == cnapol.global_policy().norm_type:
        return {"epsilon": 1e-5}

    return {}


def STEM(name=None):
    if name is None:
        counter = naming.get_uid("stem_")
        name = f"stem_{counter}"

    def apply(inputs):
        x = Conv(
            32,
            3,
            strides=2,
            **CONV_KWARGS,
            policy=cnapol.default_policy(),
            name=name + "_stem_conv_1",
        )(inputs)
        x = Norm(**norm_kwargs(), name=name + "_stem_batch_norm_1")(x)
        x = Act(name=name + "_stem_act_1")(x)

        x = Conv(32, 3, **CONV_KWARGS, name=name + "_stem_conv_2")(x)
        x = Norm(**norm_kwargs(), name=name + "_stem_batch_norm_2")(x)
        x = Act(name=name + "_stem_act_2")(x)

        x = Conv(64, 3, **CONV_KWARGS, name=name + "_stem_conv_3")(x)
        x = Norm(**norm_kwargs(), name=name + "_stem_batch_norm_3")(x)
        x = Act(name=name + "_stem_act_3")(x)

        x = Conv(64, 3, strides=2, **CONV_KWARGS, name=name + "_stem_conv_4")(x)
        x = Norm(**norm_kwargs(), name=name + "_stem_batch_norm_4")(x)
        x = Act(name=name + "_stem_act_4")(x)

        return x

    return apply


def SE(in_filters, se_ratio=0.25, expand_ratio=1, name=None):
    if name is None:
        counter = naming.get_uid("se_")
        name = f"se_{counter}"

    def apply(inputs):
        x = layers.GlobalAveragePooling2D(
            keepdims=True, name=name + "_se_squeeze"
        )(inputs)

        num_reduced_filters = max(1, int(in_filters * 4 * se_ratio))

        x = Conv(
            num_reduced_filters,
            1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "_se_reduce",
        )(x)
        x = Act()(x)

        x = layers.Conv2D(
            4 * in_filters * expand_ratio,
            1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            activation="sigmoid",
            name=name + "_se_expand",
        )(x)

        return layers.multiply([inputs, x], name=name + "_se_excite")

    return apply


def BottleneckBlock(
    filters, strides, use_projection, survival_probability=0.8, name=None
):
    if name is None:
        counter = naming.get_uid("block_0_")
        name = f"block_0_{counter}"

    def apply(inputs):
        shortcut = inputs

        if use_projection:
            filters_out = filters * 4
            shortcut = inputs
            if strides == 2:
                shortcut = layers.AveragePooling2D(
                    2,
                    2,
                    padding="same",
                    name=name + "_projection_pooling",
                )(shortcut)
            shortcut = Conv(
                filters_out, 1, **CONV_KWARGS, name=name + "_projection_conv"
            )(shortcut)
            shortcut = Norm(
                **norm_kwargs(), name=name + "_projection_batch_norm"
            )(shortcut)

        x = Conv(filters, 1, **CONV_KWARGS, name=name + "_conv_1")(inputs)
        x = Norm(**norm_kwargs(), name=name + "batch_norm_1")(x)
        x = Act(name=name + "_act_1")(x)

        x = Conv(
            filters, 3, strides=strides, **CONV_KWARGS, name=name + "_conv_2"
        )(x)
        x = Norm(**norm_kwargs(), name=name + "_batch_norm_2")(x)
        x = Act(name=name + "_act_2")(x)

        x = Conv(filters * 4, 1, **CONV_KWARGS, name=name + "_conv_3")(x)
        x = Norm(**norm_kwargs(), name=name + "_batch_norm_3")(x)

        x = SE(filters, name=name + "_se")(x)

        if survival_probability:
            x = layers.Dropout(
                survival_probability,
                noise_shape=(None, 1, 1, 1),
                name=name + "_drop",
            )(x)

        x = layers.add([x, shortcut])

        return Act(name=name + "_output_act")(x)

    return apply


def BlockGroup(
    filters, strides, num_repeats, survival_probability=0.8, name=None
):
    if name is None:
        counter = naming.get_uid("block_group_")
        name = f"block_group_{counter}"

    def apply(inputs):
        x = BottleneckBlock(
            filters=filters,
            strides=strides,
            use_projection=True,
            survival_probability=survival_probability,
            name=name + "_block_0_",
        )(inputs)

        for i in range(1, num_repeats):
            x = BottleneckBlock(
                filters=filters,
                strides=1,
                use_projection=False,
                survival_probability=survival_probability,
                name=name + f"_block_{i}_",
            )(x)
        return x

    return apply


def ResNetRS(
    depth,
    input_shape=None,
    dropout_rate=0.25,
    drop_connect_rate=0.2,
    include_top=True,
    block_args=None,
    model_name="resnet-rs",
    pooling=None,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    available_weight_variants = DEPTH_TO_WEIGHT_VARIANTS[depth]
    if weights == "imagenet":
        max_input_shape = max(available_weight_variants)
        weights = f"{weights}-i{max_input_shape}"

    weights_allow_list = [f"imagenet-i{x}" for x in available_weight_variants]
    if not (
        weights in {*weights_allow_list, None} or tf.io.gfile.exists(weights)
    ):
        raise ValueError(
            "The `weights` argument should be either `None` (random initialization), `imagenet` (pre-training on "
            "ImageNet, with highest available input shape), or the path to the weights file to be loaded. "
            f"For ResNetRS{depth} the following weight variants are available {weights_allow_list} (default=highest). "
            f"Received weights={weights}"
        )

    if weights in weights_allow_list and include_top and classes != 1000:
        raise ValueError(
            f"If using `weights` as `imagenet` or any of {weights_allow_list} with `include_top` as true, "
            f"`classes` should be 1000. Received classes={classes}"
        )

    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format="channels_last",
        require_flatten=include_top,
        weights=weights,
    )
    input_dtype = dtype_policy.dtype_policy().compute_dtype
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, dtype=input_dtype)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(
                tensor=input_tensor, shape=input_shape, dtype=input_dtype
            )
        else:
            img_input = input_tensor

    x = img_input

    if include_preprocessing:
        num_channels = input_shape[-1]
        x = layers.Rescaling(scale=1.0 / 255, name="rescale")(x)
        if num_channels == 3:
            x = layers.Normalization(
                mean=[0.485, 0.456, 0.406],
                variance=[0.229**2, 0.224**2, 0.225**2],
                name="normalize",
            )(x)

    x = STEM(name="stem_1")(x)

    if block_args is None:
        block_args = BLOCK_ARGS[depth]

    for i, args in enumerate(block_args):
        survival_probability = get_survival_probability(
            init_rate=drop_connect_rate,
            block_num=i + 2,
            total_blocks=len(block_args) + 1,
        )

        x = BlockGroup(
            filters=args["input_filters"],
            strides=(1 if i == 0 else 2),
            num_repeats=args["num_repeats"],
            survival_probability=survival_probability,
            name=f"BlockGroup{i + 2}_",
        )(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name="top_dropout")(x)

        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(
            classes, activation=classifier_activation, name="predictions"
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    if input_tensor is not None:
        inputs = operation_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = models.Model(inputs, x, name=model_name)

    if weights in weights_allow_list:
        weights_input_shape = weights.split("-")[-1]
        weights_name = f"{model_name}-{weights_input_shape}"
        if not include_top:
            weights_name += "_notop"

        filename = f"{weights_name}.h5"
        download_url = BASE_WEIGHTS_URL + filename
        weights_path = file_utils.get_file(
            fname=filename,
            origin=download_url,
            cache_subdir="models",
            file_hash=WEIGHT_HASHES[filename],
        )
        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)

    return model


def ResNetRS50(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    return ResNetRS(
        depth=50,
        include_top=include_top,
        drop_connect_rate=0.0,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-50",
        include_preprocessing=include_preprocessing,
    )


def ResNetRS101(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    return ResNetRS(
        depth=101,
        include_top=include_top,
        drop_connect_rate=0.0,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-101",
        include_preprocessing=include_preprocessing,
    )


def ResNetRS152(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    return ResNetRS(
        depth=152,
        include_top=include_top,
        drop_connect_rate=0.0,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-152",
        include_preprocessing=include_preprocessing,
    )


def ResNetRS200(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    return ResNetRS(
        depth=200,
        include_top=include_top,
        drop_connect_rate=0.1,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-200",
        include_preprocessing=include_preprocessing,
    )


def ResNetRS270(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    allow_bigger_recursion(1300)
    return ResNetRS(
        depth=270,
        include_top=include_top,
        drop_connect_rate=0.1,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-270",
        include_preprocessing=include_preprocessing,
    )


def ResNetRS350(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    allow_bigger_recursion(1500)
    return ResNetRS(
        depth=350,
        include_top=include_top,
        drop_connect_rate=0.1,
        dropout_rate=0.4,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-350",
        include_preprocessing=include_preprocessing,
    )


def ResNetRS420(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    allow_bigger_recursion(1800)
    return ResNetRS(
        depth=420,
        include_top=include_top,
        dropout_rate=0.4,
        drop_connect_rate=0.1,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-420",
        include_preprocessing=include_preprocessing,
    )


def wrap_bone_policy(model, prepr, init, channels, end_points, name):
    if (
        cnapol.default_policy().name == cnapol.global_policy().name
        or init is None
    ):
        return wrap_bone(model, prepr, init, channels, end_points, name)

    with cnapol.policy_scope(cnapol.default_policy()):
        base_model = wrap_bone(model, prepr, init, channels, end_points, name)

    base_weights = {w.path: w for w in base_model.weights}
    if len(base_model.weights) != len(base_weights.keys()):
        raise ValueError("Some weights have equal names")

    ext_model = wrap_bone(model, prepr, None, channels, end_points, name)

    ext_weights = []
    for random_weight in ext_model.weights:
        weight_name = random_weight.path

        if weight_name not in base_weights:
            ext_weights.append(random_weight)
            continue

        if random_weight.shape != base_weights[weight_name].shape:
            ext_weights.append(random_weight)
            continue

        ext_weights.append(base_weights[weight_name])

    ext_model.set_weights(ext_weights)
    ext_model.trainable = True

    return ext_model


BACKBONES.register("resnet_rs_50")(
    (
        partial(wrap_bone_policy, ResNetRS50, None),
        [
            None,
            "stem_1_stem_act_3",
            "BlockGroup2__block_2__output_act",
            "BlockGroup3__block_3__output_act",
            "BlockGroup4__block_5__output_act",
            "BlockGroup5__block_2__output_act",
        ],
    )
)

BACKBONES.register("resnet_rs_101")(
    (
        partial(wrap_bone_policy, ResNetRS101, None),
        [
            None,
            "stem_1_stem_act_3",
            "BlockGroup2__block_2__output_act",
            "BlockGroup3__block_3__output_act",
            "BlockGroup4__block_22__output_act",
            "BlockGroup5__block_2__output_act",
        ],
    )
)

BACKBONES.register("resnet_rs_152")(
    (
        partial(wrap_bone_policy, ResNetRS152, None),
        [
            None,
            "stem_1_stem_act_3",
            "BlockGroup2__block_2__output_act",
            "BlockGroup3__block_7__output_act",
            "BlockGroup4__block_35__output_act",
            "BlockGroup5__block_2__output_act",
        ],
    )
)

BACKBONES.register("resnet_rs_200")(
    (
        partial(wrap_bone_policy, ResNetRS200, None),
        [
            None,
            "stem_1_stem_act_3",
            "BlockGroup2__block_2__output_act",
            "BlockGroup3__block_23__output_act",
            "BlockGroup4__block_35__output_act",
            "BlockGroup5__block_2__output_act",
        ],
    )
)

BACKBONES.register("resnet_rs_270")(
    (
        partial(wrap_bone_policy, ResNetRS270, None),
        [
            None,
            "stem_1_stem_act_3",
            "BlockGroup2__block_3__output_act",
            "BlockGroup3__block_28__output_act",
            "BlockGroup4__block_52__output_act",
            "BlockGroup5__block_3__output_act",
        ],
    )
)

BACKBONES.register("resnet_rs_350")(
    (
        partial(wrap_bone_policy, ResNetRS350, None),
        [
            None,
            "stem_1_stem_act_3",
            "BlockGroup2__block_3__output_act",
            "BlockGroup3__block_35__output_act",
            "BlockGroup4__block_71__output_act",
            "BlockGroup5__block_3__output_act",
        ],
    )
)

BACKBONES.register("resnet_rs_420")(
    (
        partial(wrap_bone_policy, ResNetRS420, None),
        [
            None,
            "stem_1_stem_act_3",
            "BlockGroup2__block_3__output_act",
            "BlockGroup3__block_43__output_act",
            "BlockGroup4__block_86__output_act",
            "BlockGroup5__block_3__output_act",
        ],
    )
)


def wrap_bone_stride8(model, prepr, init, channels, end_points, name):
    base_model = wrap_bone_policy(
        model, prepr, init, channels, end_points, name
    )

    ext_config = base_model.get_config()

    stride_patches = []
    for layer in ext_config["layers"]:
        if "kernel_size" not in layer["config"]:
            continue
        if layer["config"]["kernel_size"] in {1, (1, 1)}:
            continue

        if layer["config"]["name"].startswith("BlockGroup4__"):
            dilation = 2
        elif layer["config"]["name"].startswith("BlockGroup5__"):
            dilation = 4
        else:
            continue

        if layer["config"]["strides"] not in {1, (1, 1)}:
            if layer["config"]["strides"] not in {2, (2, 2)}:
                raise ValueError("Unexpected strides.")
            dilation //= 2

        stride_patches.append((layer["config"]["name"], dilation))

    for layer, dilation in stride_patches:
        ext_config = patch_config(ext_config, [layer], "strides", 1)
        ext_config = patch_config(
            ext_config, [layer], "dilation_rate", dilation
        )

    for i, layer in enumerate(ext_config["layers"]):
        if layer["config"]["name"] not in {
            "BlockGroup4__block_0__projection_pooling",
            "BlockGroup5__block_0__projection_pooling",
        }:
            continue

        layer["class_name"] = "Activation"
        layer["config"]["activation"] = "linear"
        del layer["config"]["pool_size"]
        del layer["config"]["padding"]
        del layer["config"]["strides"]
        del layer["config"]["data_format"]

        ext_config["layers"][i] = layer

    ext_model = models.Model.from_config(ext_config)
    ext_model.set_weights(base_model.get_weights())
    ext_model.trainable = True

    return ext_model


BACKBONES.register("resnet_rs_50_s8")(
    (
        partial(wrap_bone_stride8, ResNetRS50, None),
        [
            None,
            "stem_1_stem_act_3",
            "BlockGroup2__block_2__output_act",
            "BlockGroup3__block_3__output_act",
            "BlockGroup4__block_5__output_act",
            "BlockGroup5__block_2__output_act",
        ],
    )
)
