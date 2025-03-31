import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 3  # Update this to match your actual number of categories
TEST_SIZE = 0.4

def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Check if labels are empty
    if len(labels) == 0:
        sys.exit("Error: No labels found after loading data. Check your data directory.")

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Debugging output
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]  # Use the provided filename from command-line arguments
        model.save(filename)
        print(f"Model saved to {filename}.")

def load_data(images_dir):
    img_width, img_height = IMG_WIDTH, IMG_HEIGHT
    images, labels = [], []
    
    # Iterate over each category directory
    for i in range(NUM_CATEGORIES):
        category_dir = os.path.join(images_dir, str(i))
        
        # Check if category directory exists
        if not os.path.exists(category_dir):
            print(f"Category directory {category_dir} does not exist.")
            continue
        
        # Check if there are any images in the category directory
        if not os.listdir(category_dir):
            print(f"Category directory {category_dir} is empty.")
            continue
        
        # Iterate over each image file in the category directory
        for filename in os.listdir(category_dir):
            img_path = os.path.join(category_dir, filename)
            img = cv2.imread(img_path)
            
            # Check if the image was loaded successfully
            if img is None:
                print(f"Failed to load image {img_path}. Skipping.")
                continue
            
            img = cv2.resize(img, (img_width, img_height))
            images.append(img)
            labels.append(i)

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Debugging output
    print(f"Loaded {len(images)} images and {len(labels)} labels.")
    
    # Check if labels are empty
    if len(labels) == 0:
        raise ValueError("No labels found. Please check the images directory.")

    return images, labels

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Define the model architecture
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Return the model
    return model

if __name__ == "__main__":
    main()