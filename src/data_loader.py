# src/data_loader.py
"""
Dataset helpers. Stronger augmentation defaults and target_size reduced to 128x128 by default.
"""
import os
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dataset_structure(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    for split in ("train", "test"):
        p = os.path.join(data_dir, split)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Missing folder: {p}. Expected train/ and test/ subfolders.")
        for cls in ("O", "R"):
            cls_p = os.path.join(p, cls)
            if not os.path.isdir(cls_p):
                raise FileNotFoundError(f"Missing class folder: {cls_p}")

def create_generators(data_dir,
                      target_size=(128,128),
                      batch_size=64,
                      validation_split=0.2,
                      seed=42):
    """
    Returns: train_gen, val_gen, test_gen, class_indices
    Uses stronger data augmentation to reduce overfitting.
    """
    check_dataset_structure(data_dir)
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='reflect',
        validation_split=validation_split
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=seed
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=True,
        seed=seed
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=1,
        class_mode='binary',
        shuffle=False
    )

    class_indices = train_gen.class_indices
    logger.info(f"Class indices: {class_indices}")
    return train_gen, val_gen, test_gen, class_indices
