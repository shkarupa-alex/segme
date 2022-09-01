import tensorflow as tf
from keras import models, layers


def patch_config(config, path, param, patch):
    if 'layers' not in config:
        raise ValueError('Can\'t find layers in config {}'.format(config))

    head, tail = path[0], path[1:]
    for i in range(len(config['layers'])):
        found = isinstance(head, int) and i == head or config['layers'][i]['config'].get('name', None) == head
        if not found:
            continue

        if tail:
            config['layers'][i]['config'] = patch_config(config['layers'][i]['config'], tail, param, patch)
        elif param not in config['layers'][i]['config']:
            raise ValueError('Parameter {} not found in layer {}'.format(param, head))
        else:
            patched = patch if not callable(patch) else patch(config['layers'][i]['config'][param])
            config['layers'][i]['config'][param] = patched

        return config

    top_layers = [layer.get('name', None) for layer in config['layers']]

    raise ValueError('Layer {} not found. Top-level layers: {}'.format(head, top_layers))


def get_layer(model, name_idx):
    if not isinstance(name_idx, str):
        return model.get_layer(index=name_idx).output

    if ' > ' not in name_idx:
        return model.get_layer(name=name_idx).output

    name_parts = name_idx.split(' > ')
    head_name, tail_name = name_parts[0], ' > '.join(name_parts[1:])
    child = model.get_layer(head_name)
    if not hasattr(child, 'get_layer'):
        raise ValueError('Could not obtain layer {} from node {}'.format(name_idx, child))

    return get_layer(child, tail_name)


def wrap_bone(model, prepr, init, channels, end_points):
    input_image = layers.Input(name='image', shape=(None, None, channels))

    if prepr is not None:
        input_prep = layers.Lambda(prepr, name='preprocess')(input_image)
    else:
        input_prep = input_image

    base_model = model(input_tensor=input_prep, include_top=False, weights=init)
    output_feats = [get_layer(base_model, name_idx) for name_idx in end_points]

    down_stack = models.Model(inputs=input_image, outputs=output_feats)
    down_stack.trainable = init is None

    return down_stack
