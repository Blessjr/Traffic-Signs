def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    # Loop through each category
    for label in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(label))
        
        # Loop through each image file in the category directory
        for filename in os.listdir(category_dir):
            if filename.endswith('.ppm'):  # Assuming the images are in .ppm format
                img_path = os.path.join(category_dir, filename)
                
                # Read and resize the image
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                
                images.append(img)
                labels.append(label)

    return (np.array(images), np.array(labels))