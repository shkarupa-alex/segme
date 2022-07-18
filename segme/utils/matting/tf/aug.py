import tensorflow as tf


def augment_inverse(image, prob=0.05, seed=None):
    with tf.name_scope('augment_foreground'):
        image = tf.convert_to_tensor(image, 'uint8')

        if 4 != image.shape.rank:
            raise ValueError('Expecting `alpha` rank to be 4.')

        if 3 != image.shape[-1]:
            raise ValueError('Expecting `foreground` channels size to be 3.')

        if 'uint8' != image.dtype:
            raise ValueError('Expecting `foreground` dtype to be `uint8`.')

        batch = tf.shape(image)[0]
        apply = tf.random.uniform([batch, 1, 1, 1], 0., 1., seed=seed)
        apply = tf.cast(apply < prob, image.dtype)

        return (255 - image) * apply + image * (1 - apply)


def augment_alpha(alpha, prob=0.3, seed=None):
    with tf.name_scope('augment_alpha'):
        alpha = tf.convert_to_tensor(alpha, 'uint8')

        if 4 != alpha.shape.rank:
            raise ValueError('Expecting `alpha` rank to be 4.')

        if 1 != alpha.shape[-1]:
            raise ValueError('Expecting `alpha` channels size to be 1.')

        if 'uint8' != alpha.dtype:
            raise ValueError('Expecting `alpha` dtype to be `uint8`.')

        batch = tf.shape(alpha)[0]
        apply, gamma_switch, gamma, alpha_switch = tf.split(
            tf.random.uniform([batch, 1, 1, 4], 0., 1., seed=seed), 4, axis=-1)

        apply = tf.cast(apply < prob, alpha.dtype)

        gamma_switch = tf.cast(gamma_switch > 0.5, gamma_switch.dtype)
        gamma = (gamma / 2 + 0.5) * gamma_switch + (gamma + 1.) * (1 - gamma_switch)

        orig_dtype = alpha.dtype
        alpha_ = alpha if orig_dtype in {tf.float16, tf.float32} else tf.image.convert_image_dtype(alpha, 'float32')

        alpha_switch = tf.cast(alpha_switch > 0.5, alpha_switch.dtype)
        alpha_ = tf.pow(alpha_, gamma) * alpha_switch + (1 - tf.pow(1 - alpha_, gamma)) * (1 - alpha_switch)

        return tf.image.convert_image_dtype(alpha_, orig_dtype, saturate=True) * apply + alpha * (1 - apply)


def augment_trimap(trimap, prob=0.1, seed=None):
    with tf.name_scope('augment_trimap'):
        trimap = tf.convert_to_tensor(trimap, 'uint8')

        if 4 != trimap.shape.rank:
            raise ValueError('Expecting `trimap` rank to be 4.')

        if 1 != trimap.shape[-1]:
            raise ValueError('Expecting `trimap` channels size to be 1.')

        if 'uint8' != trimap.dtype:
            raise ValueError('Expecting `trimap` dtype to be `uint8`.')

        batch = tf.shape(trimap)[0]
        apply = tf.random.uniform([batch, 1, 1, 1], 0., 1., seed=seed)
        apply = tf.cast(apply < prob, trimap.dtype)

        fg = tf.convert_to_tensor(255, 'uint8')
        un = tf.convert_to_tensor(128, 'uint8')

        return tf.where(tf.equal(trimap, fg), un, trimap) * apply + trimap * (1 - apply)
