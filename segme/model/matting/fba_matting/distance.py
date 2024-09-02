from keras.src import ops
from segme.ops import saturate_cast
from tfmiss.image import euclidean_distance


def distance_transform(trimap, length=320):
    clicks = []
    for value in [0, 255]:
        twomap = ops.cast(trimap != value, "uint8") * 255
        distance = -ops.square(euclidean_distance(twomap))
        clicks.extend(
            [
                ops.exp(distance / (2 * (0.02 * length) ** 2)),
                ops.exp(distance / (2 * (0.08 * length) ** 2)),
                ops.exp(distance / (2 * (0.16 * length) ** 2)),
            ]
        )

    clicks = ops.concatenate(clicks, axis=-1)
    clicks = saturate_cast(ops.round(clicks * 255.0), dtype="uint8")

    return clicks
