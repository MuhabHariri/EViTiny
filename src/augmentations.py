import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

target_height = 256
target_width = 256
IMG_SIZE = 256
CROP_HEIGHT = 155
CROP_WIDTH = 155


def random_shear(image):
    with tf.device('/cpu:0'):
        shear_angle = tf.random.uniform([], minval=-0.25, maxval=0.25)
        replace = tf.constant([0, 0, 0], dtype=tf.float32)
        image = tfa.image.shear_x(image, shear_angle, replace)
        image = tfa.image.shear_y(image, shear_angle, replace)
    return image

def random_translate(image):
    height, width, _ = image.shape
    width_shift = width * 0.25
    height_shift = height * 0.25
    width_shift_range = tf.random.uniform([], -width_shift, width_shift)
    height_shift_range = tf.random.uniform([], -height_shift, height_shift)
    translations = [width_shift_range, height_shift_range]
    return tfa.image.translate(image, translations)

def random_zoom(image):
    zoom_factor = tf.random.uniform(shape=[], minval=0.75, maxval=1.25)
    new_height = tf.cast(tf.multiply(tf.constant(target_height, dtype=tf.float32), zoom_factor), tf.int32)
    new_width = tf.cast(tf.multiply(tf.constant(target_width, dtype=tf.float32), zoom_factor), tf.int32)
    image = tf.image.resize(image, [new_height, new_width])
    image = tf.image.resize_with_crop_or_pad(image, target_height, target_width)
    return image

def random_flip(image):
    return tf.image.random_flip_left_right(image)

def random_rotate(image):
    min_angle = -20 * np.pi / 180
    max_angle = 20 * np.pi / 180
    angle = tf.random.uniform([], minval=min_angle, maxval=max_angle)
    return tfa.image.rotate(image, angle)

def random_adjust_brightness(image):
    delta = tf.random.uniform([], minval=0.06, maxval=0.14)
    return tf.image.adjust_brightness(image, delta)

def random_channel_shift(image):
    delta = tf.random.uniform([], minval=-50, maxval=50) / 255.0
    return tf.clip_by_value(image + delta, 0.0, 1.0)

def random_saturation(image):
    saturation_factor = tf.random.uniform([], minval=1, maxval=1.2)
    return tf.image.adjust_saturation(image, saturation_factor)

def random_contrast(image):
    contrast_factor = tf.random.uniform([], minval=1.2, maxval=1.6)
    return tf.image.adjust_contrast(image, contrast_factor)

def custom_random_translate(image, width_shift_range, height_shift_range):
    width_shift = tf.random.uniform([], width_shift_range[0], width_shift_range[1])
    height_shift = tf.random.uniform([], height_shift_range[0], height_shift_range[1])
    translations = [width_shift, height_shift]
    return tfa.image.translate(image, translations)

def combined_random_augmentations_with_order(image):
    height, width, _ = image.shape
    width_shift_range = [width * 0.1, width * 0.25]
    height_shift_range = [height * 0.1, height * 0.25]

    augmentations = [
        lambda img: tf.image.adjust_brightness(img, tf.random.uniform([], 0.06, 0.2)),
        lambda img: tf.clip_by_value(img + tf.random.uniform([], -80, 80) / 255.0, 0.0, 1.0),
        lambda img: tf.image.adjust_saturation(img, tf.random.uniform([], 1, 1.6)),
        lambda img: tf.image.adjust_contrast(img, tf.random.uniform([], 1.2, 1.9)),
        lambda img: custom_random_translate(img, width_shift_range, height_shift_range),
    ]

    np.random.shuffle(augmentations)
    for augment in augmentations:
        image = augment(image)
    return image

def random_crop_and_pad(image):
    crop_size = [CROP_HEIGHT, CROP_WIDTH, 3]
    image = tf.image.random_crop(image, size=crop_size)
    _ = target_height - CROP_HEIGHT
    _ = target_width - CROP_WIDTH
    image = tf.image.pad_to_bounding_box(
        image, offset_height=0, offset_width=0,
        target_height=target_height, target_width=target_width
    )
    return image

def random_black_rectangle(image):
    height, width, _ = tf.unstack(tf.shape(image))
    dtype = image.dtype
    rectangle_height = tf.random.uniform([], minval=45, maxval=60, dtype=tf.int32)
    rectangle_width = tf.random.uniform([], minval=80, maxval=100, dtype=tf.int32)
    start_y = tf.random.uniform([], minval=0, maxval=height - rectangle_height, dtype=tf.int32)
    start_x = tf.random.uniform([], minval=0, maxval=width - rectangle_width, dtype=tf.int32)

    mask = tf.ones((height, width, 1), dtype=dtype)
    mask = tf.tensor_scatter_nd_update(
        mask,
        indices=tf.reshape(tf.stack(tf.meshgrid(
            tf.range(start_y, start_y + rectangle_height),
            tf.range(start_x, start_x + rectangle_width),
            indexing='ij'
        ), axis=-1), (-1, 2)),
        updates=tf.zeros((rectangle_height * rectangle_width, 1), dtype=dtype)
    )
    mask = tf.tile(mask, [1, 1, 3])
    return image * mask


def _ops_sequence():
    F  = random_flip
    C0 = combined_random_augmentations_with_order
    Rb = random_black_rectangle
    Sh = random_shear
    Cr = random_crop_and_pad
    Tr = random_translate
    Ro = random_rotate
    Br = random_adjust_brightness
    Zo = random_zoom
    Ch = random_channel_shift

    s1 = [F, C0, Rb, Sh, Cr, Tr, Ro, Rb, Br, Zo, C0, Ch]
    s2 = [F, Rb, Sh, Cr, Tr, Ro, C0, Rb, Br, Zo, Ch, Cr, C0]
    s3 = [F, Rb, Sh, Cr, Tr, C0, Ro, Rb, Br, Zo, Ch, C0]
    s4 = [F, Rb, Sh, Cr, Tr, Ro, C0, Rb, Br, Cr, Zo, C0]
    s5 = [F, Rb, Sh, Cr, Tr, C0, Ro, Rb, Br, Zo, Cr, C0]

    return s1 + s2 + s3 + s4 + s5 


def augment(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    total_augmentations = 1
    augmentations_applied = 0
    ops = _ops_sequence()  

    for op in ops:
        rnd = tf.random.uniform(())
        if rnd < 0.2 and augmentations_applied < total_augmentations:
            image = op(image)
            augmentations_applied += 1

    return image, label

