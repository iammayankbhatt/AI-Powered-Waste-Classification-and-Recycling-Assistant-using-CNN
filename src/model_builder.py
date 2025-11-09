# src/model_builder.py
"""
Improved model builder:
- build_cnn: small-from-scratch CNN using GlobalAveragePooling2D + L2 + dropout
- build_transfer: MobileNetV2 frozen-top transfer-learning model
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model   #type: ignore
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout)  #type: ignore
from tensorflow.keras.regularizers import l2  #type: ignore
from tensorflow.keras.applications import MobileNetV2   #type: ignore

def build_cnn(input_shape=(128,128,3), dropout_rate=0.5, l2_rate=1e-4):
    """Lightweight CNN using GlobalAveragePooling to avoid huge dense layer."""
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, (3,3), activation='relu', padding='same', kernel_regularizer=l2(l2_rate)),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(l2_rate)),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2(l2_rate)),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        GlobalAveragePooling2D(),               # MUCH fewer params than Flatten()
        Dense(256, activation='relu', kernel_regularizer=l2(l2_rate)),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid', dtype='float32')  # float32 safe if mixed precision used
    ])
    return model

def build_transfer(input_shape=(128,128,3), dropout_rate=0.5, l2_rate=1e-4, backbone="MobileNetV2"):
    """
    Transfer learning using MobileNetV2 (imagenet) as a frozen base.
    Returns a compiled Model (not compiled here so the caller can choose optimizer).
    """
    if backbone != "MobileNetV2":
        raise ValueError("Only MobileNetV2 implemented in this helper.")
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_rate))(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, activation='sigmoid', dtype='float32')(x)

    model = Model(inputs=base.input, outputs=out)
    return model
