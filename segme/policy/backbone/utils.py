import numpy as np
from keras.src import backend
from keras.src import layers
from keras.src import models

from segme.common.inguard import InputGuard


def patch_config(config, path, param, patch):
    if "layers" not in config:
        raise ValueError(f"Can't find layers in config {config}")

    head, tail = path[0], path[1:]
    for i in range(len(config["layers"])):
        found = (
            isinstance(head, int)
            and i == head
            or config["layers"][i]["config"].get("name", None) == head
        )
        if not found:
            continue

        if tail:
            config["layers"][i]["config"] = patch_config(
                config["layers"][i]["config"], tail, param, patch
            )
        elif param not in config["layers"][i]["config"]:
            raise ValueError(f"Parameter {param} not found in layer {head}")
        else:
            patched = (
                patch
                if not callable(patch)
                else patch(config["layers"][i]["config"][param])
            )
            config["layers"][i]["config"][param] = patched

        return config

    top_layers = [layer.get("name", None) for layer in config["layers"]]

    raise ValueError(f"Layer {head} not found. Top-level layers: {top_layers}")


def patch_channels(model, mean=None, variance=None):
    if not isinstance(model, models.Model):
        raise ValueError(
            f"Expecting model to be an instance of `keras.Model`. "
            f"Got: {type(model)}"
        )
    config = model.get_config()

    if "layers" not in config:
        raise ValueError(f"Can't find layers in config {config}")
    if not config["layers"]:
        raise ValueError(f"Layers are empty in config {config}")

    guard = None
    for i, layer in enumerate(config["layers"]):
        if "__input_guard__" == layer["config"]["name"]:
            guard = i
            break
    if guard is None:
        raise ValueError(f"Input guard layer not found in config {config}")

    if guard + 1 == len(config["layers"]):
        raise ValueError("Input guard should not be the last layer in model")

    recipient = config["layers"][guard + 1]
    if (
        1 != len(recipient["inbound_nodes"])
        or 1 != len(recipient["inbound_nodes"][0]["args"])
        or "__input_guard__"
        != recipient["inbound_nodes"][0]["args"][0]["config"]["keras_history"][
            0
        ]
    ):
        raise ValueError(
            f"Expecting input guard layer to be connected with next one. "
            f"Got: {recipient}"
        )

    input_shape = tuple(config["layers"][guard]["build_config"]["input_shape"])
    source_shape = tuple(
        config["layers"][guard + 1]["build_config"]["input_shape"]
    )

    config["layers"][guard + 1]["build_config"]["input_shape"] = input_shape
    config["layers"][guard + 1]["inbound_nodes"] = config["layers"][guard][
        "inbound_nodes"
    ]

    for i, layer in enumerate(config["layers"]):
        if i < guard + 2:
            continue

        if (
            "__input_guard__"
            in layer["inbound_nodes"][0]["args"][0]["config"]["keras_history"]
        ):
            raise ValueError(
                f"Expecting input guard to be connected only with one layer. "
                f"Got: {layer}"
            )

        if source_shape != layer["build_config"]["input_shape"]:
            break
        config["layers"][i]["build_config"]["input_shape"] = input_shape
        config["layers"][i]["inbound_nodes"][0]["args"][0]["config"][
            "shape"
        ] = input_shape

        if "Normalization" == layer.get("class_name", None):
            add_channels = max(0, input_shape[-1] - 3)
            if mean is None:
                raise ValueError(
                    f"Expecting mean to have {add_channels} values. "
                    f"Got: {mean}"
                )
            elif isinstance(mean, float):
                mean = [mean] * add_channels
            elif add_channels != len(mean):
                raise ValueError(
                    f"Expecting mean to have {add_channels} values. "
                    f"Got: {mean}"
                )
            mean = list(mean)

            if variance is None:
                raise ValueError(
                    f"Expecting variance to have {add_channels} values. "
                    f"Got: {variance}"
                )
            elif isinstance(variance, float):
                variance = [variance] * add_channels
            elif add_channels != len(variance):
                raise ValueError(
                    f"Expecting variance to have {add_channels} values. "
                    f"Got: {variance}"
                )
            variance = list(variance)

            config["layers"][i]["config"]["mean"] = (
                config["layers"][i]["config"]["mean"][: input_shape[-1]] + mean
            )
            config["layers"][i]["config"]["variance"] = (
                config["layers"][i]["config"]["variance"][: input_shape[-1]]
                + variance
            )

    config["layers"].pop(guard)

    weights = {w.path: w for w in model.weights}
    if len(model.weights) != len(weights.keys()):
        raise ValueError("Some weights have equal names")

    ext_model = models.Model.from_config(config)

    ext_weights = []
    for random_weight in ext_model.weights:
        weight_name = random_weight.path

        if weight_name not in weights:
            raise ValueError(
                f"Model weight {weight_name} not found after deserialization"
            )
        base_weight = backend.convert_to_numpy(weights[weight_name])
        random_weight = backend.convert_to_numpy(random_weight)

        if len(random_weight.shape) != len(base_weight.shape):
            raise ValueError(
                f"Model weight {weight_name} rank is changed "
                f"after deserialization"
            )

        if random_weight.shape == base_weight.shape:
            ext_weights.append(base_weight)
            continue

        if (
            random_weight.shape[:2] != base_weight.shape[:2]
            or random_weight.shape[3:] != base_weight.shape[3:]
        ):
            raise ValueError(
                f"Unexpected weight shape difference: "
                f"{random_weight.shape} vs {base_weight.shape}"
            )

        diff = random_weight.shape[2] - base_weight.shape[2]
        if -2 == diff:
            extended_weight = base_weight.sum(axis=2, keepdims=True)
        elif -1 == diff:
            extended_weight = base_weight.sum(axis=2, keepdims=True)
            extended_weight = np.concatenate(
                [extended_weight, random_weight[:, :, :1]], axis=2
            )
        elif diff > 0:
            extended_weight = np.concatenate(
                [base_weight, random_weight[:, :, 3:]], axis=2
            )
        else:
            raise ValueError(
                f"Expecting weight difference to be -2, -1 or greater then "
                f"0. Got {random_weight.shape} vs {base_weight.shape}"
            )

        ext_weights.append(extended_weight)

    ext_model.set_weights(ext_weights)
    ext_model.trainable = True

    return ext_model


def get_layer(model, name_idx):
    if not isinstance(name_idx, str):
        return model.get_layer(index=name_idx).output

    if " > " not in name_idx:
        return model.get_layer(name=name_idx).output

    name_parts = name_idx.split(" > ")
    head_name, tail_name = name_parts[0], " > ".join(name_parts[1:])
    child = model.get_layer(head_name)
    if not hasattr(child, "get_layer"):
        raise ValueError(
            "Could not obtain layer {} from node {}".format(name_idx, child)
        )

    return get_layer(child, tail_name)


def wrap_bone(model, prepr, init, end_points, name, input_tensor=None):
    if input_tensor is None:
        input_tensor = layers.Input(
            name="image", shape=(None, None, 3), dtype="uint8"
        )

    if prepr is None:
        x = input_tensor
    elif "tf" == prepr:
        x = layers.Rescaling(
            scale=1.0 / 127.5,
            offset=-1,
            name="rescale",
        )(input_tensor)
    elif "torch" == prepr:
        x = layers.Normalization(
            mean=np.array([0.485, 0.456, 0.406], "float32") * 255.0,
            variance=(np.array([0.229, 0.224, 0.225], "float32") * 255.0) ** 2,
            name="normalize",
        )(input_tensor)
    else:
        raise ValueError("Unsupported preprocessing")

    if 3 != x.shape[-1]:
        x = InputGuard(name="__input_guard__")(x)

    base_model = model(input_tensor=x, include_top=False, weights=init)
    output_feats = [get_layer(base_model, name_idx) for name_idx in end_points]

    down_stack = models.Model(
        inputs=base_model.inputs, outputs=tuple(output_feats), name=name
    )
    down_stack.trainable = init is None

    return down_stack
