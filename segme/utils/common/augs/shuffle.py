import tensorflow as tf
from segme.utils.common.augs.common import apply, validate


def shuffle(image, masks, weight, prob, perm=None, name=None):
    with tf.name_scope(name or 'shuffle'):
        return apply(
            image, masks, weight, prob,
            lambda x: _shuffle(x, perm), tf.identity, tf.identity)


def _shuffle(image, perm=None, name=None):
    with tf.name_scope(name or 'shuffle_'):
        image, _, _ = validate(image, None, None)

        if perm is not None:
            perm = tf.convert_to_tensor(perm, 'int32', name='perm')
            if 1 != perm.shape.rank:
                raise ValueError('Expecting `perm` rank to be 1.')
            image = tf.gather(image, perm, batch_dims=-1)
        else:
            image = tf.transpose(image, [3, 0, 1, 2])
            image = tf.random.shuffle(image)
            image = tf.transpose(image, [1, 2, 3, 0])

        return image
