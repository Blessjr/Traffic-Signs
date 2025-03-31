import sys
import cv2
import numpy as np
import tensorflow as tf

# Load the model
model_path = sys.argv[1]  # Use the first command-line argument as the model filename
model = tf.keras.models.load_model(model_path)

# Function to prepare input image
def prepare_image(image_path):
    print(f"Loading image from: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image. Check if the file exists and is accessible.")
        raise ValueError(f"Image at path '{image_path}' could not be loaded. Please check the path.")
    img = cv2.resize(img, (30, 30))  # Resize to the input size expected by the model
    img = img.astype('float32') / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Example image prediction
if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: python predict_sign.py model.h5 image_path")

    image_path = sys.argv[2]  # Use the second command-line argument as the image filename
    prepared_image = prepare_image(image_path)
    prediction = model.predict(prepared_image)
    predicted_class = np.argmax(prediction, axis=1)

    print(f"Predicted class: {predicted_class[0]}")