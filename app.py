import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from inference_sdk import InferenceHTTPClient
import streamlit as st
import os

# Retrieve the API key from Streamlit secrets
API_KEY = st.secrets["ROBOFLOW_API_KEY"]

# Initialize Roboflow clients
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key=API_KEY
)

CLIENT2 = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key=API_KEY
)

# Define class names for prediction
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'] # Modify this list according to your model

def create_mask_from_points(image_shape, points):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    points_array = np.array([[p['x'], p['y']] for p in points], dtype=np.int32)
    cv2.fillPoly(mask, [points_array], 1)
    return mask

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    segmentation_result = CLIENT.infer(img, model_id="plant-disease-detection-ryzqa/7")
    infected_area_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    total_leaf_area_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    if predicted_class != 'Potato___healthy':
        segmentation_predictions = segmentation_result['predictions']
        for seg_pred in segmentation_predictions:
            if seg_pred['confidence'] > 0.4:
                points = seg_pred['points']
                single_mask = create_mask_from_points(img.shape, points)
                infected_area_mask = cv2.bitwise_or(infected_area_mask, single_mask)

    leaf_segmentation_result = CLIENT2.infer(img, model_id="segmentasi-daun/4")
    leaf_segmentation_predictions = leaf_segmentation_result['predictions']
    for leaf_seg_pred in leaf_segmentation_predictions:
        if leaf_seg_pred['confidence'] > 0.4:
            leaf_points = leaf_seg_pred['points']
            leaf_single_mask = create_mask_from_points(img.shape, leaf_points)
            total_leaf_area_mask = cv2.bitwise_or(total_leaf_area_mask, leaf_single_mask)

    infected_area_pixels = np.count_nonzero(infected_area_mask)
    total_leaf_area_pixels = np.count_nonzero(total_leaf_area_mask)

    if total_leaf_area_pixels > 0 and predicted_class != 'Potato___healthy':
        infected_area_percentage = ((infected_area_pixels / total_leaf_area_pixels)) * 100 + 5
    else:
        infected_area_percentage = 0
    # Cap the infected area percentage at 90%
    infected_area_percentage = min(infected_area_percentage, 90)
    return predicted_class, confidence, infected_area_mask, total_leaf_area_mask, infected_area_percentage

# Streamlit app UI
st.title('Potato Disease Detection and Segmentation')

st.write(
    "Upload an image of a potato leaf for disease detection and segmentation. The app will predict the disease class, "
    "confidence, and segment infected areas of the leaf."
)

# File uploader for user to upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    img = Image.open(uploaded_file)
    img = img.convert("RGB")  # Ensure it's in RGB format
    img_np = np.array(img)

    # Display uploaded image
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Load the model from the same directory as app.py
    model_path = os.path.join(os.getcwd(), "potatoes.h5")  # Path to your model in the same directory as app.py
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Make predictions
    predicted_class, confidence, infected_area_mask, total_leaf_area_mask, infected_area_percentage = predict(model, img_np)

    # Display results
    st.subheader(f"Prediction: {predicted_class}")
    st.subheader(f"Confidence: {confidence}%")
    st.subheader(f"Infected Area: {infected_area_percentage:.2f}%")

    # Display combined infected area and leaf area mask with color overlay
    st.subheader("Combined Mask with Infected Area (Red) and Leaf Area (Green)")

    # Create a color mask for the infected area (Red)
    color_mask_infected = np.zeros_like(img_np, dtype=np.uint8)
    color_mask_infected[infected_area_mask > 0] = [255, 0, 0]  # Red color for the infected area

    # Create a color mask for the total leaf area (Green)
    color_mask_leaf = np.zeros_like(img_np, dtype=np.uint8)
    color_mask_leaf[total_leaf_area_mask > 0] = [0, 255, 0]  # Green color for the leaf area

    # Remove green areas where red exists (to prevent blending into yellow)
    color_mask_leaf[infected_area_mask > 0] = [0, 0, 0]  # Set green mask to black where red is already present

    # Combine both masks by layering them
    combined_image = np.where(color_mask_infected > 0, color_mask_infected, img_np)  # Red areas over the original image
    combined_image = np.where(color_mask_leaf > 0, color_mask_leaf, combined_image)  # Green areas over the previous result

    # Display the combined image with both color masks
    st.image(combined_image, caption="Combined Infected and Leaf Area Mask", use_container_width=True)

else:
    st.write("Please upload an image to start.")



