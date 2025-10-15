# src/data_pipeline.py
import os
import glob
import pathlib
import numpy as np
import tensorflow as tf

DEFAULT_IMAGE_SHAPE = (256, 256)

def build_class_names(train_dir: str) -> np.ndarray:
    """Discover class folders under train_dir (same logic as your script)."""
    return np.array([
        p.name for p in pathlib.Path(train_dir).glob('*')
        if p.is_dir() and p.name != "LICENSE.txt"
    ])

def list_train_filepaths(train_dir: str) -> list[str]:
    """List all image paths in class subfolders under train_dir."""
    return glob.glob(os.path.join(train_dir, '*', '*'))

def count_images(root: str) -> int:
    """Count images in root/class_x/*"""
    return len(glob.glob(os.path.join(root, '*', '*')))

def load_and_preprocess_image(path: tf.Tensor, image_shape=DEFAULT_IMAGE_SHAPE) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_shape)
    image = image / 255.0
    return image

def get_label(file_path: tf.Tensor, class_names: tf.Tensor) -> tf.Tensor:
    """Return one-hot bool vector as in your original get_label."""
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == class_names  # bool one-hot

def load_and_preprocess_data(file_path: tf.Tensor,
                             class_names: tf.Tensor,
                             image_shape=DEFAULT_IMAGE_SHAPE):
    image = load_and_preprocess_image(file_path, image_shape)
    label = get_label(file_path, class_names)
    return image, label

def make_train_dataset(file_paths: list[str],
                       class_names: np.ndarray,
                       batch_size: int,
                       augment_fn,  # function(image, label) -> (image, label)
                       image_shape=DEFAULT_IMAGE_SHAPE,
                       shuffle_buffer: int = 10000) -> tf.data.Dataset:
    class_names_tf = tf.constant(class_names)

    def map_fn(fp):
        return load_and_preprocess_data(fp, class_names_tf, image_shape)

    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    ds = ds.shuffle(buffer_size=min(shuffle_buffer, len(file_paths)))
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if augment_fn is not None:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def make_val_dataset(val_dir: str,
                     class_names: np.ndarray,
                     batch_size: int,
                     image_shape=DEFAULT_IMAGE_SHAPE) -> tf.data.Dataset:
    class_names_tf = tf.constant(class_names)

    def map_fn(fp):
        return load_and_preprocess_data(fp, class_names_tf, image_shape)

    ds = tf.data.Dataset.list_files(os.path.join(val_dir, '*', '*'))
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
