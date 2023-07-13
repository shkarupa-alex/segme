import itertools
import tensorflow as tf
from keras.mixed_precision import global_policy
from segme.common.shape import get_shape


def grid_sample(features, grid, mode='bilinear', align_corners=False, symmetric_pad=False):
    if mode not in {'bilinear', 'nearest'}:
        raise ValueError('Wrong interpolation mode. Only "bilinear" and "nearest" supported')

    features_shape, _ = get_shape(features, axis=[0, 1, 2])
    features_size = features_shape[1:3]
    (batch_size, point_height, point_width), _ = get_shape(grid, axis=[0, 1, 2])

    assertions = [
        tf.debugging.assert_equal(
            features_shape[0], batch_size, message='Batch size should be the same for features and grid'),
        tf.debugging.assert_greater_equal(
            tf.reduce_min(grid), tf.cast(-1.0, grid.dtype), message='Grid values should be in range [-1; 1]'),
        tf.debugging.assert_less_equal(
            tf.reduce_max(grid), tf.cast(1.0, grid.dtype), message='Grid values should be in range [-1; 1]')]
    with tf.control_dependencies(assertions):
        pad_mode = 'SYMMETRIC' if symmetric_pad else 'CONSTANT'
        safe_features = tf.pad(features, [[0, 0], [1, 1], [1, 1], [0, 0]], mode=pad_mode)
        safe_features = tf.cast(safe_features, grid.dtype)
        grid = tf.reverse(grid, axis=[-1])
        size = tf.cast(features_size, grid.dtype)

        if align_corners:
            grid = (grid + 1.) * (size - 1) * 0.5
        else:
            grid = (grid + 1.) * size * 0.5 - 0.5

        batch_idx = tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1, 1))
        coord_batches = tf.tile(batch_idx, (1, point_height, point_width, 1))
        coord_bounds = tf.cast(features_size, 'int32') + 1

        def _lookup(coords):
            coords = tf.clip_by_value(tf.cast(coords, 'int32') + 1, 0, coord_bounds)
            indices = tf.concat([coord_batches, coords], axis=-1)
            return tf.gather_nd(safe_features, indices)

        if 'bilinear' == mode:
            grid_nw = tf.math.floor(grid)
            grid_ne = grid_nw + [1, 0]
            grid_sw = grid_nw + [0, 1]
            grid_se = grid_nw + [1, 1]

            nw = tf.math.reduce_prod(grid_se - grid, axis=-1, keepdims=True)
            ne = tf.math.reduce_prod((grid_sw - grid) * [1, -1], axis=-1, keepdims=True)
            sw = tf.math.reduce_prod((grid_ne - grid) * [-1, 1], axis=-1, keepdims=True)
            se = tf.math.reduce_prod(grid - grid_nw, axis=-1, keepdims=True)

            result = tf.add_n([
                _lookup(grid_nw) * nw,
                _lookup(grid_ne) * ne,
                _lookup(grid_sw) * sw,
                _lookup(grid_se) * se])

        else:  # 'nearest' == mode
            result = _lookup(tf.math.round(grid))

        features_dtype = tf.dtypes.as_dtype(features.dtype)
        if features_dtype.is_integer:
            result = tf.round(result)

        return tf.cast(result, features.dtype)


def make_coords(inputs, dtype=None):
    dtype = dtype or global_policy().compute_dtype

    if isinstance(inputs, (tuple, list)):
        if 3 != len(inputs):
            raise ValueError(f'Expected inputs to be a list or tuple with batch/height/width values, got {inputs}')
        batch, height, width = inputs
    else:
        inputs = tf.convert_to_tensor(inputs)
        (batch, height, width), _ = get_shape(inputs, axis=[0, 1, 2])

    height_ = 1. / tf.cast(height, dtype)
    width_ = 1. / tf.cast(width, dtype)

    vertical = height_ - 1. + 2 * height_ * tf.cast(tf.range(height, dtype='float32'), dtype)
    horizontal = width_ - 1. + 2 * width_ * tf.cast(tf.range(width, dtype='float32'), dtype)

    mesh = tf.meshgrid(vertical, horizontal, indexing='ij')
    join = tf.stack(mesh, axis=-1)
    outputs = tf.tile(join[None], [batch, 1, 1, 1])

    if not isinstance(inputs, (tuple, list)):
        outputs.set_shape(inputs.shape[:-1] + [2])

    return outputs


def query_features(features, coords, imnet, posnet=None, cells=None, feat_unfold=False, local_ensemble=True,
                   symmetric_pad=False):
    """
    Proposed in "Learning Continuous Image Representation with Local Implicit Image Function"
    https://arxiv.org/abs/2012.09161
    """
    features = tf.convert_to_tensor(features)
    if 4 != features.shape.rank:
        raise ValueError('Features must have rank 4')

    coords = tf.convert_to_tensor(coords)
    if 4 != coords.shape.rank:
        raise ValueError('Coordinates must have rank 4')

    if cells is not None:
        cells = tf.convert_to_tensor(cells)
        if 4 != cells.shape.rank:
            raise ValueError('Coordinates must have rank 4')

    dtype = features.dtype
    coords = tf.cast(coords, dtype)

    cell_decode = cells is not None

    if feat_unfold:
        if not isinstance(feat_unfold, int) or feat_unfold < 3:
            raise ValueError('Unfold kernel size must be an integer greater or equal to 3')
        if not feat_unfold % 2:
            raise ValueError('Unfold kernel size must be odd')
        features = tf.image.extract_patches(
            features, [1, feat_unfold, feat_unfold, 1], [1] * 4, [1] * 4, padding='SAME')

    if local_ensemble:
        vxvy = itertools.product([-1., 1.], [-1., 1.])
        epsilon = 1e-6
    else:
        vxvy = [(0, 0)]
        epsilon = 0.

    h_w, _ = get_shape(features, axis=[1, 2])
    h_w = tf.cast(h_w, dtype)
    rx_ry = 1. / h_w

    feat_coords = make_coords(features)

    if cell_decode:
        rel_cells = tf.cast(cells, dtype) * h_w

    preds, areas = [], []
    for vx_vy in vxvy:
        coords_ = coords + (vx_vy * rx_ry + epsilon)
        coords_ = tf.clip_by_value(coords_, -1 + 1e-6, 1 - 1e-6)
        coords_ = tf.reverse(coords_, axis=[-1])

        q_feats = grid_sample(features, coords_, mode='nearest', align_corners=False, symmetric_pad=symmetric_pad)
        q_coords = grid_sample(feat_coords, coords_, mode='nearest', align_corners=False, symmetric_pad=symmetric_pad)
        rel_coords = (coords - q_coords) * h_w

        if posnet is not None:
            queries = [q_feats, posnet(rel_coords)]
        else:
            queries = [q_feats, rel_coords]
        if cell_decode:
            queries.append(rel_cells)
        queries = tf.concat(queries, axis=-1)

        pred = imnet(queries)
        preds.append(pred)

        if local_ensemble:
            area = tf.cast(rel_coords, 'float32')
            area = tf.reduce_prod(area, axis=-1, keepdims=True)
            areas.insert(0, tf.abs(area) + 1e-9)

    if local_ensemble:
        outputs = [tf.cast(pred, 'float32') * area for pred, area in zip(preds, areas)]
        outputs = tf.math.add_n(outputs) / tf.math.add_n(areas)
        outputs = tf.saturate_cast(outputs, dtype)
    else:
        outputs = preds[0]

    return outputs
