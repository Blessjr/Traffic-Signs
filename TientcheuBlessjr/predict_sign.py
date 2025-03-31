import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf

# Load the model
model_path = r'C:\Users\Bless\Desktop\Artificial-Intelligence\Traffic\TientcheuBlessjr\best_model.h5'  # Use raw string
model = tf.keras.models.load_model(model_path)

def prepare_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be loaded.")
    img = cv2.resize(img, (30, 30))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict():
    image_path = filedialog.askopenfilename()
    if not image_path:
        return

    try:
        prepared_image = prepare_image(image_path)
        prediction = model.predict(prepared_image)
        predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get the predicted class index

        # Map the predicted index to the exact value
        if predicted_class_index == 0:
            predicted_value = 0
        elif predicted_class_index == 1:
            predicted_value = 1
        elif predicted_class_index == 2:
            predicted_value = 2
        else:
            predicted_value = "Unknown"  # Handle unexpected values

        messagebox.showinfo("Prediction Result", f"Predicted value: {predicted_value}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the GUI
root = tk.Tk()
root.title("Traffic Sign Predictor")

predict_button = tk.Button(root, text="Select Image", command=predict)
predict_button.pack(pady=20)

root.mainloop()