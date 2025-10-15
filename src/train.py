# src/train.py
import os
import csv
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping

from .augmentations import augment
from .model import create_autoencoder, remove_spatial_dropout, EViTiny
from .data_pipeline import (
    build_class_names,
    list_train_filepaths,
    count_images,
    make_train_dataset,
    make_val_dataset,
)

# ------------------- Config -------------------
IMAGE_SHAPE = (256, 256)
BATCH_SIZE = 16
INPUT_SHAPE = (256, 256, 3)
LEARNING_RATE = 0.0002
ENCODER_WEIGHTS_PATH = 'Encoder_Weights.h5'

# ------------------- Strategy + tf.data helpers (Windows-safe) -------------------
def _make_strategy():
    """
    On Windows, NCCL is unavailable; use HierarchicalCopyAllReduce.
    Elsewhere, default MirroredStrategy is fine.
    Fallback to OneDeviceStrategy if MirroredStrategy fails.
    """
    try:
        if os.name == "nt":
            return tf.distribute.MirroredStrategy(
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
            )
        return tf.distribute.MirroredStrategy()
    except Exception as e:
        print(f"[WARN] MirroredStrategy failed ({e}). Falling back to OneDeviceStrategy.")
        gpus = tf.config.list_physical_devices("GPU")
        device = "/gpu:0" if gpus else "/cpu:0"
        return tf.distribute.OneDeviceStrategy(device=device)

def _apply_dist_options(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Make sharding explicit to silence auto-shard warnings."""
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    return ds.with_options(options)

# ------------------- Path input helpers -------------------
def _clean_user_path(p: str) -> str:
    p = p.strip()

    # r"...." or R'....'
    if (p.startswith(('r"', "r'", 'R"', "R'")) and (p.endswith('"') or p.endswith("'"))):
        p = p[2:-1]
    # "...." or '....'
    elif (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1]
    # rC:\...  or RC:\... (no quotes)
    elif (p.startswith(('r', 'R')) and len(p) > 1 and (p[1] == ':' or p[1] in ('\\', '/'))):
        p = p[1:]

    p = os.path.expanduser(os.path.expandvars(p))
    p = os.path.abspath(os.path.normpath(p))
    return p

def _ask_path(prompt_text: str) -> str:
    raw = input(prompt_text)
    p = _clean_user_path(raw)
    if not os.path.isdir(p):
        alt = os.path.abspath(os.path.normpath(p.replace('\\', '/')))
        if os.path.isdir(alt):
            return alt
        raise FileNotFoundError(f"Directory not found: {raw}\nResolved path: {p}")
    return p

def _ask_int(prompt_text: str) -> int:
    return int(input(prompt_text).strip())

# ------------------- Visualization (preserved behavior) -------------------
def _plot_and_save_images(images, labels, class_names, epoch, save_dir="plots"):
    import pathlib

    plt.figure(figsize=(12, 8))
    for i, (image, label) in enumerate(zip(images, labels)):
        if i == 8:
            break
        plt.subplot(2, 4, i + 1)
        plt.imshow(np.clip(image, 0.0, 1.0))
        idx = int(np.argmax(label))
        name = class_names[idx]
        if isinstance(name, (bytes, bytearray)):
            name = name.decode("utf-8")
        plt.title(name)
        plt.axis('off')
    os.makedirs(save_dir, exist_ok=True)
    out = pathlib.Path(save_dir) / f"epoch_{epoch}_augmented_images.png"
    plt.savefig(out)
    plt.close() 

# ------------------- Public entrypoint -------------------
def run_training():
    # Interactive inputs
    train_dir = _ask_path("Enter path to TRAINING data directory (e.g., /data/train): ")
    val_dir = _ask_path("Enter path to VALIDATION data directory (e.g., /data/val): ")
    user_num_classes = _ask_int("Enter number of classes: ")
    num_epochs = _ask_int("Enter number of epochs: ")

    # Discover classes from folders (same logic as original)
    class_names = build_class_names(train_dir)
    found_num_classes = len(class_names)
    if user_num_classes != found_num_classes:
        print(f"[INFO] You entered {user_num_classes} classes, but {found_num_classes} were found in '{train_dir}'. Using {found_num_classes}.")
    num_classes = found_num_classes

    # File paths & counts
    file_paths = list_train_filepaths(train_dir)
    random.shuffle(file_paths)
    train_samples = len(file_paths)
    val_samples = count_images(val_dir)

    train_steps_per_epoch = int(np.ceil(train_samples / BATCH_SIZE))
    val_steps_per_epoch = int(np.ceil(val_samples / BATCH_SIZE))

    # Strategy
    strategy = _make_strategy()
    replicas = strategy.num_replicas_in_sync

    # Essential training info
    print("=== Training Configuration ===")
    print(f"Train images: {train_samples} | Val images: {val_samples}")
    print(f"Batch size: {BATCH_SIZE} | Epochs: {num_epochs}")
    print(f"Steps/epoch: {train_steps_per_epoch} | Val steps: {val_steps_per_epoch}")
    print(f"Optimizer: Adam | Learning rate: {LEARNING_RATE}")
    print(f"TF Distributed: MirroredStrategy with {replicas} replicas")

    # Validation dataset (build once)
    val_data = make_val_dataset(val_dir, class_names, BATCH_SIZE, image_shape=IMAGE_SHAPE)
    val_data = _apply_dist_options(val_data)

    # CSV header (preserved)
    csv_file = 'training_log.csv'
    with open(csv_file, mode='w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'accuracy', 'val_accuracy', 'loss', 'val_loss'])

    # Build model within strategy
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        autoencoder = create_autoencoder(input_shape=INPUT_SHAPE)
        # identical slice point as your script
        encoder = models.Model(inputs=autoencoder.input, outputs=autoencoder.layers[15].output)

        if not os.path.isfile(ENCODER_WEIGHTS_PATH):
            raise FileNotFoundError(
                f"Required encoder weights file not found: {ENCODER_WEIGHTS_PATH}. "
                "Place it in the repo root or update ENCODER_WEIGHTS_PATH in src/train.py."
            )
        encoder.load_weights(ENCODER_WEIGHTS_PATH)

        encoder_no_dropout = remove_spatial_dropout(encoder)

        inputs = tf.keras.Input(shape=INPUT_SHAPE)
        x = encoder_no_dropout(inputs)
        output = EViTiny(
            x, embedding_size=256, num_heads_list=3,
            num_classes=num_classes, num_transformer_layers=1
        )
        model = models.Model(inputs=inputs, outputs=output)
        print(f"Total parameters: {model.count_params():,}")

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

    # Early stopping (preserved settings)
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=25,
        verbose=0,
        restore_best_weights=False
    )

    # Epoch-by-epoch loop (same pattern as original: epochs=1 inside loop)
    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        random.shuffle(file_paths)
        train_data = make_train_dataset(
            file_paths=file_paths,
            class_names=class_names,
            batch_size=BATCH_SIZE,
            augment_fn=augment,
            image_shape=IMAGE_SHAPE,
            shuffle_buffer=10000
        )
        train_data = _apply_dist_options(train_data)

        # Preview/save augmented batch like your script
        for image_batch, label_batch in train_data.take(1):
            _plot_and_save_images(image_batch.numpy(), label_batch.numpy(), class_names, epoch)

        history = model.fit(
            train_data,
            epochs=1,
            validation_data=val_data,
            steps_per_epoch=train_steps_per_epoch,
            validation_steps=val_steps_per_epoch,
            callbacks=[early_stopping],
            verbose=1
        )

        # Manual CSV logging (preserved)
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                history.history['accuracy'][0],
                history.history['val_accuracy'][0],
                history.history['loss'][0],
                history.history['val_loss'][0]
            ])

        os.makedirs('Saved_Weights', exist_ok=True)
        model.save_weights(os.path.join('Saved_Weights', f'model_weights_epoch_{epoch + 1}.h5'))

# Optional convenience: allow running this module directly
if __name__ == "__main__":
    run_training()
