
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

# Class names
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Load trained CNN model
@st.cache_resource
def load_cnn_model():
    model_path = Path(__file__).parent / "models" / "best_cnn.keras"
    return tf.keras.models.load_model(
        model_path,
        compile=False
    )

# Application heading
st.title("CNN Image Classification")
st.write(
    "Upload an image and classify it using "
    "a Convolutional Neural Network trained on CIFAR-10."
)

st.info("This model can recognize 10 different image categories.")

# Load model
with st.spinner("Loading CNN model..."):
    model = load_cnn_model()

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocess image
    resized_image = image.resize((32, 32))

    image_array = np.array(
        resized_image,
        dtype="float32"
    ) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Predict
    if st.button("Classify Image", type="primary"):

        with st.spinner("Analyzing image..."):
            prediction = model.predict(
                image_array,
                verbose=0
            )

        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

        label = class_names[predicted_class]

        # Display results
        st.success(f"Predicted Category: {label}")

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence * 100:.2f}%"
        )

        st.progress(confidence)

        # Show all class probabilities
        st.subheader("Class Probabilities")

        probabilities = {
            class_names[i]: float(prediction[0][i])
            for i in range(len(class_names))
        }

        st.bar_chart(probabilities)

st.divider()
st.caption(
    "CNN Image Classification | "
    "TensorFlow | Keras | CIFAR-10"
)