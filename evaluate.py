
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Create results folder
os.makedirs("results", exist_ok=True)

# Define class names
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Load the test dataset
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize test images
x_test = x_test.astype("float32") / 255.0

# Load trained model
model = tf.keras.models.load_model("models/best_cnn.keras")

print("Trained model loaded successfully!")

# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Generate predictions
predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)
y_true = y_test.flatten()

# Generate classification report
report = classification_report(
    y_true,
    y_pred,
    labels=range(10),
    target_names=class_names,
    zero_division=0
)

print("\nClassification Report:\n")
print(report)

# Save classification report
with open("results/classification_report.txt", "w") as file:
    file.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
    file.write(f"Test Loss: {test_loss:.4f}\n\n")
    file.write(report)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=range(10))

fig, ax = plt.subplots(figsize=(12, 10))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45
)

plt.title("CNN Image Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.close()

print("\nClassification report saved successfully!")
print("Confusion matrix saved successfully!")
print("Model evaluation completed successfully!")