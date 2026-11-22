
import os
import tensorflow as tf
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

# Define class names
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Select test image
image_path = "test.jpg"

if not os.path.isfile(image_path):
    print("Test image not found!")
    raise SystemExit

# Load trained model
print("Loading trained model...", flush=True)

model = tf.keras.models.load_model(
    "models/best_cnn.keras",
    compile=False
)

print("Model loaded successfully!")

# Load and preprocess image
image = Image.open(image_path).convert("RGB")
resized_image = image.resize((32, 32))

image_array = np.array(
    resized_image, dtype="float32"
) / 255.0

image_array = np.expand_dims(image_array, axis=0)

# Make prediction
prediction = model.predict(image_array, verbose=0)

predicted_class = np.argmax(prediction[0])
confidence = np.max(prediction[0]) * 100
label = class_names[predicted_class]

# Display prediction in terminal
print("\nPrediction Result")
print("Predicted Class:", label)
print(f"Confidence Score: {confidence:.2f}%")

# Save prediction result
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title(
    f"Prediction: {label}\n"
    f"Confidence Score: {confidence:.2f}%"
)

plt.axis("off")
plt.tight_layout()
plt.savefig("results/prediction.png", dpi=150)
plt.close()

print("Prediction image saved successfully!")
print("Image classification completed!")
