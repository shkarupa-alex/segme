import itertools
import tensorflow as tf
from keras import backend
from .gridsample import grid_sample


def make_coords(inputs, dtype=None):
    inputs = tf.convert_to_tensor(inputs, dtype=dtype)
    batch, height, width, _ = tf.unstack(tf.shape(inputs))

    if dtype is None:
        dtype = backend.floatx()

    height_ = 1. / tf.cast(height, dtype)
    width_ = 1. / tf.cast(width, dtype)

    vertical = height_ - 1. + 2 * height_ * tf.cast(tf.range(height, dtype='float32'), dtype)
    horizontal = width_ - 1. + 2 * width_ * tf.cast(tf.range(width, dtype='float32'), dtype)

    mesh = tf.meshgrid(vertical, horizontal, indexing='ij')
    join = tf.stack(mesh, axis=-1)
    outputs = tf.tile(join[None], [batch, 1, 1, 1])

    outputs.set_shape(inputs.shape[:-1] + [2])

    return outputs


def query_features(features, coords, imnet, cells=None, feat_unfold=True, local_ensemble=True, dtype=None):
    """
    Proposed in "Learning Continuous Image Representation with Local Implicit Image Function"
    https://arxiv.org/pdf/2012.09161.pdf
    """
    if dtype is None:
        dtype = backend.floatx()

    features = tf.cast(features, dtype)
    coords = tf.cast(coords, dtype)

    cell_decode = cells is not None

    if feat_unfold:
        features = tf.image.extract_patches(features, [1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], padding='SAME')

    if local_ensemble:
        vxvy = itertools.product([-1., 1.], [-1., 1.])
        epsilon = 1e-6
    else:
        vxvy = [(0, 0)]
        epsilon = 0.

    h_w = tf.cast(tf.shape(features)[1:3], dtype)
    rx_ry = 1. / h_w

    feat_coords = make_coords(features, dtype)

    if cell_decode:
        rel_cells = cells * h_w

    preds, areas = [], []
    for vx_vy in vxvy:
        coords_ = coords + (vx_vy * rx_ry + epsilon)
        coords_ = tf.clip_by_value(coords_, -1 + 1e-6, 1 - 1e-6)
        coords_ = tf.reverse(coords_, axis=[-1])

        q_feats = grid_sample(features, coords_, mode='nearest', align_corners=False)
        q_coords = grid_sample(feat_coords, coords_, mode='nearest', align_corners=False)
        rel_coords = (coords - q_coords) * h_w

        queries = [q_feats, rel_coords]
        if cell_decode:
            queries.append(rel_cells)
        queries = tf.concat(queries, axis=-1)

        pred = imnet(queries)
        preds.append(tf.cast(pred, 'float32'))

        if local_ensemble:
            area = tf.cast(rel_coords, 'float32')
            area = tf.reduce_prod(area, axis=-1, keepdims=True)
            areas.insert(0, tf.abs(area) + 1e-9)

    if local_ensemble:
        outputs = [pred * area for pred, area in zip(preds, areas)]
        outputs = tf.math.add_n(outputs) / tf.math.add_n(areas)
    else:
        outputs = preds[0]

    return outputs
