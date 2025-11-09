# src/train.py
"""
Training script with improved defaults:
- target size 128x128
- batch size 64
- workers + use_multiprocessing
- saves class_indices->models/class_indices.json
- option to choose transfer learning (MobileNetV2) or build_cnn
"""
import os
import json
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam   #type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau   #type: ignore
from src.data_loader import create_generators  # type: ignore
from src.model_builder import build_cnn, build_transfer  # type: ignore

def detect_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU detected:", gpus)
    else:
        print("No GPU detected. Training on CPU.")

def train(data_dir='data',
          model_path='models/waste_classifier.h5',
          epochs=12,
          batch_size=64,
          target_size=(128,128),
          use_transfer=False,
          workers=4):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs('models', exist_ok=True)

    detect_gpu()

    train_gen, val_gen, test_gen, class_indices = create_generators(
        data_dir, target_size=target_size, batch_size=batch_size
    )

    # Save class mapping for predict-time
    class_map_path = os.path.join(os.path.dirname(model_path), "class_indices.json")
    with open(class_map_path, "w") as f:
        json.dump(class_indices, f)
    print("Saved class indices to:", class_map_path)

    if use_transfer:
        print("Building transfer model (MobileNetV2)...")
        model = build_transfer(input_shape=(target_size[0], target_size[1], 3), dropout_rate=0.5)
        optimizer = Adam(learning_rate=1e-4)
    else:
        print("Building from-scratch CNN...")
        model = build_cnn(input_shape=(target_size[0], target_size[1], 3), dropout_rate=0.5)
        optimizer = Adam(learning_rate=1e-3)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, early, reduce_lr],
        workers=workers,
        use_multiprocessing=True
    )

    # Evaluate best saved model on test_set
    if os.path.exists(model_path):
        print("Loading best saved model for final evaluation:", model_path)
        best = tf.keras.models.load_model(model_path)
        test_loss, test_acc = best.evaluate(test_gen, verbose=1)
        print(f"Final test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")
    else:
        print("Model path not found after training; skipping final eval.")

    return history, class_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--model_path', default='models/waste_classifier.h5')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--target_size', type=int, nargs=2, default=(128,128))
    parser.add_argument('--transfer', action='store_true', help='use MobileNetV2 transfer learning')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    train(args.data_dir, args.model_path, args.epochs, args.batch_size, tuple(args.target_size), args.transfer, args.workers)
