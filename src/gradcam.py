# src/gradcam.py
import numpy as np
import tensorflow as tf
import cv2

def compute_gradcam(model, img_array, last_conv_layer_name=None, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given image and model.
    img_array: preprocessed image array (1, H, W, 3)
    """
    # find last conv layer if not provided
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        raise ValueError("No Conv2D layer found in model.")

    # Use model.inputs (not [model.inputs]) to avoid nested-list warning
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index_tensor = tf.argmax(predictions[0])
            pred_index = int(pred_index_tensor.numpy())
        # get the score for the target class
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # weighted combination of maps
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # ReLU then normalize
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap.numpy())
    heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_gradcam(heatmap, image, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 1 - alpha, heatmap_color, alpha, 0)
    return overlay
