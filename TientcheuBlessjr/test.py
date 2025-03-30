# Step 1: Implement the load_data function

import os
import cv2
import numpy as np

def load_data(data_dir):
    # Define the image dimensions
    img_width, img_height = IMG_WIDTH, IMG_HEIGHT
    
    # Initialize the lists to store images and labels
    images, labels = [], []
    
    # Iterate over each category directory
    for i in range(42):
        # Construct the path to the category directory
        category_dir = os.path.join(data_dir, str(i))
        
        # Iterate over each image file in the category directory
        for filename in os.listdir(category_dir):
            # Construct the path to the image file
            img_path = os.path.join(category_dir, filename)
            
            # Read the image using OpenCV
            img = cv2.imread(img_path)
            
            # Resize the image to the specified dimensions
            img = cv2.resize(img, (img_width, img_height))
            
            # Add the image and label to the lists
            images.append(img)
            labels.append(i)
    
    # Convert the lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Return the images and labels
    return images, labels


# Step 2: Implement the get_model function

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def get_model():
    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(42, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Return the model
    return model

#Evaluate neural network performance
model.evaluate(x_test, y_test, verbose=2)


filename = "best_model.h5"
model.save(filename)