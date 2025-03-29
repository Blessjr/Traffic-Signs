import os
import cv2
import numpy as np

IMG_WIDTH, IMG_HEIGHT = 30, 30  # Set your desired dimensions
NUM_CATEGORIES = 43  # Number of traffic sign categories

def load_data(data_dir):
    images = []
    labels = []

    # Traverse through each category directory
    for label in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(label))
        
        # Read each image in the category
        for filename in os.listdir(category_dir):
            if filename.endswith('.ppm'):  # Use appropriate file extension
                img_path = os.path.join(category_dir, filename)
                
                # Read and resize the image
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(img)
                labels.append(label)

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return (images, labels)