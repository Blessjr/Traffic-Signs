import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
import tensorflow as tf

# Load the model (make sure to adjust the path to your model)
model = tf.keras.models.load_model('your_model.h5')

IMG_WIDTH, IMG_HEIGHT = 30, 30  # Same dimensions used in traffic.py

def load_and_prepare_image(image_path):
    """Load and preprocess the image."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def classify_image():
    """Classify the uploaded image."""
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Load and prepare the image
    img = load_and_prepare_image(file_path)

    # Perform the prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])  # Get the index of the highest probability

    # Show the result
    messagebox.showinfo("Prediction Result", f"Predicted Class: {predicted_class}")

# Create the main application window
root = tk.Tk()
root.title("Traffic Sign Recognition")

# Create and place a button to upload an image
upload_button = tk.Button(root, text="Upload Image", command=classify_image)
upload_button.pack(pady=20)

# Start the application
root.mainloop()