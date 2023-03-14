import tensorflow as tf
from itertools import permutations
from segme.utils.common.augs.common import apply, validate


def shuffle(image, masks, weight, prob, perm=None, name=None):
    with tf.name_scope(name or 'shuffle'):
        return apply(
            image, masks, weight, prob,
            lambda x: _shuffle(x, perm),
            lambda x: tf.identity(x),
            lambda x: tf.identity(x))


def _shuffle(image, perm=None, name=None):
    with tf.name_scope(name or 'shuffle_'):
        image, _, _ = validate(image, None, None)

        if perm is not None:
            perm = tf.convert_to_tensor(perm, 'int32', name='perm')
            if 1 != perm.shape.rank:
                raise ValueError('Expecting `perm` rank to be 2.')
        else:
            perms = list(permutations(range(image.shape[-1])))
            perms = tf.convert_to_tensor(perms, 'int32', name='perms')
            idx = tf.random.uniform([], maxval=len(perms), dtype='int32')
            perm = perms[idx]

        image = tf.gather(image, perm, batch_dims=-1)

        return image
