import tensorflow as tf


def adjust_brightness(x, delta):
    return tf.image.adjust_brightness(x, delta)


def adjust_contrast(x, factor):
    return tf.image.adjust_contrast(x, factor)


def adjust_gamma(x, gamma=1, gain=1):
    return tf.image.adjust_gamma(x, gamma=gamma, gain=gain)


def adjust_hue(x, delta):
    return tf.image.adjust_hue(x, delta)


def adjust_jpeg_quality(x, quality):
    return tf.image.adjust_jpeg_quality(x, quality)


def adjust_saturation(x, factor):
    return tf.image.adjust_saturation(x, factor)


def grayscale_to_rgb(x):
    return tf.image.grayscale_to_rgb(x)


def histogram_fixed_width(x, x_range, nbins=100):
    return tf.histogram_fixed_width(x, x_range, nbins=nbins)
