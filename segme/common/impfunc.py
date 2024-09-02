import itertools

from keras import ops
from keras.src import backend

from segme.ops import extract_patches
from segme.ops import grid_sample
from segme.ops import saturate_cast


def make_coords(batch, height, width, dtype=None, name=None):
    with backend.name_scope(name or "make_coords"):
        dtype = dtype or backend.floatx()

        height_ = 1.0 / ops.cast(height, dtype)
        width_ = 1.0 / ops.cast(width, dtype)

        vertical = height_ - 1.0 + 2 * height_ * ops.arange(height, dtype=dtype)
        horizontal = width_ - 1.0 + 2 * width_ * ops.arange(width, dtype=dtype)

        mesh = ops.meshgrid(vertical, horizontal, indexing="ij")
        join = ops.stack(mesh, axis=-1)
        outputs = ops.tile(join[None], [batch, 1, 1, 1])

        return outputs


def query_features(
    features,
    coords,
    imnet,
    posnet=None,
    cells=None,
    feat_unfold=False,
    local_ensemble=True,
    name=None,
):
    """
    Proposed in "Learning Continuous Image Representation with Local Implicit
    Image Function"
    https://arxiv.org/abs/2012.09161
    """
    with backend.name_scope(name or "query_features"):
        features = backend.convert_to_tensor(features)
        if 4 != features.shape.rank:
            raise ValueError("Features must have rank 4")

        coords = backend.convert_to_tensor(coords)
        if 4 != coords.shape.rank:
            raise ValueError("Coordinates must have rank 4")

        if cells is not None:
            cells = backend.convert_to_tensor(cells)
            if 4 != cells.shape.rank:
                raise ValueError("Coordinates must have rank 4")

        dtype = features.dtype
        coords = ops.cast(coords, dtype)

        cell_decode = cells is not None

        if feat_unfold:
            if not isinstance(feat_unfold, int) or feat_unfold < 3:
                raise ValueError(
                    "Unfold kernel size must be an integer "
                    "greater or equal to 3"
                )
            if not feat_unfold % 2:
                raise ValueError("Unfold kernel size must be odd")
            features = extract_patches(
                features,
                [feat_unfold, feat_unfold],
                [1, 1],
                [1, 1],
                padding="same",
            )

        if local_ensemble:
            vxvy = itertools.product([-1.0, 1.0], [-1.0, 1.0])
            epsilon = 1e-6
        else:
            vxvy = [(0, 0)]
            epsilon = 0.0

        batch, height, width, _ = ops.shape(features)
        h_w = ops.cast([height, width], dtype)
        rx_ry = 1.0 / h_w

        feat_coords = make_coords(batch, height, width, dtype=features.dtype)

        if cell_decode:
            rel_cells = ops.cast(cells, dtype) * h_w

        preds, areas = [], []
        for vx_vy in vxvy:
            coords_ = coords + (vx_vy * rx_ry + epsilon)
            coords_ = ops.clip(coords_, -1 + 1e-6, 1 - 1e-6)
            coords_ = ops.flip(coords_, axis=-1)

            q_feats = grid_sample(features, coords_, mode="nearest")
            q_coords = grid_sample(feat_coords, coords_, mode="nearest")
            rel_coords = (coords - q_coords) * h_w

            if posnet is not None:
                queries = [q_feats, posnet(rel_coords)]
            else:
                queries = [q_feats, rel_coords]
            if cell_decode:
                queries.append(rel_cells)
            queries = ops.concatenate(queries, axis=-1)

            pred = imnet(queries)
            preds.append(pred)

            if local_ensemble:
                area = ops.cast(rel_coords, "float32")
                area = ops.prod(area, axis=-1, keepdims=True)
                areas.insert(0, ops.abs(area) + 1e-9)

        if local_ensemble:
            outputs = [
                ops.cast(pred, "float32") * area
                for pred, area in zip(preds, areas)
            ]
            outputs = sum(outputs) / sum(areas)

            outputs = saturate_cast(outputs, dtype)
        else:
            outputs = preds[0]

        return outputs
