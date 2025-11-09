# src/app_streamlit.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  #type: ignore
from PIL import Image
import cv2
import os
from gradcam import compute_gradcam, overlay_gradcam
from predict import preprocess_image, load_class_map

# --- Configuration ---
MODEL_PATH = "models/waste_classifier_finetuned.h5"  # use your fine-tuned model
TARGET_SIZE = (128, 128)
st.set_page_config(page_title="♻️ Waste Classifier", layout="centered")

st.title("♻️ Smart Waste Classification with CNN + Grad-CAM")

# Load model once
@st.cache_resource
def load_trained_model(path):
    model = load_model(path)
    return model

model = load_trained_model(MODEL_PATH)
class_map = load_class_map(model_dir=os.path.dirname(MODEL_PATH))

st.markdown("Upload an image of waste (Organic or Recyclable) to classify it:")
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    # st.image(img, caption="Uploaded Image", use_column_width=True) depriciated now,
    st.image(img, caption="Uploaded Image", use_container_width=True)


    # Preprocess for model
    img_resized = img.resize(TARGET_SIZE)
    arr = np.array(img_resized) / 255.0
    arr_input = np.expand_dims(arr, axis=0)

    # Predict
    preds = model.predict(arr_input)
    score = float(preds[0][0])
    label_idx = int(round(score))
    label = class_map.get(label_idx, "Unknown")

    if score >= 0.5:
        st.success(f"♻️ Predicted: **Recyclable** (score={score:.3f})")
    else:
        st.warning(f"🌱 Predicted: **Organic** (score={score:.3f})")

    # Compute Grad-CAM
    with st.spinner("Generating Grad-CAM..."):
        heatmap = compute_gradcam(model, arr_input)
        overlay = overlay_gradcam(heatmap, np.array(img))
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    st.subheader("🔍 Grad-CAM Explanation")

    st.image(overlay_rgb, caption="Highlighted regions influencing prediction", use_container_width=True)


    

