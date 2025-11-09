# src/predict.py
"""
Predict CLI and small helper. Loads class_indices.json (if available) to map indices to labels.
"""
import os
import argparse
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
import json
import sys

def load_class_map(model_dir='models'):
    path = os.path.join(model_dir, "class_indices.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            d = json.load(f)
        # invert mapping {label: idx} -> {idx: label}
        inv = {int(v): k for k, v in d.items()}
        return inv
    return {0: "O", 1: "R"}

def preprocess_image(img_path, target_size=(128,128)):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def find_model_path(preferred="models/waste_classifier_finetuned.h5"):
    # prefer preferred, then search for .keras/.h5
    if os.path.exists(preferred):
        return preferred
    # also check .keras
    alt = preferred.replace(".h5", ".keras")
    if os.path.exists(alt):
        return alt
    # search repo
    for root, _, files in os.walk(".", topdown=True):
        for f in files:
            if f.endswith(".h5") or f.endswith(".keras"):
                return os.path.join(root, f)
    raise FileNotFoundError("No saved model (.h5 or .keras) found. Train model first.")

def predict_image(model_path, image_path, target_size=(128,128), threshold=0.5):
    model = load_model(model_path)
    arr = preprocess_image(image_path, target_size=target_size)
    pred = model.predict(arr)[0][0]
    idx = int(round(pred))
    class_map = load_class_map(model_dir=os.path.dirname(model_path) or "models")
    label = class_map.get(idx, "Unknown")
    score = float(pred)
    return label, score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', default='models/waste_classifier.h5')
    args = parser.parse_args()

    model_path = find_model_path(preferred=args.model)
    label, score = predict_image(model_path, args.image)
    print(f"Model: {model_path}")
    print(f"Predicted label: {label}  (sigmoid score: {score:.4f})")
    if score >= 0.5:
        print("Interpreted as: Recyclable")
    else:
        print("Interpreted as: Organic")

if __name__ == "__main__":
    main()
