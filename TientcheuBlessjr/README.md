# Traffic Sign Prediction System

In the realm of advanced technology, the ability to predict traffic signs through machine learning has emerged as a significant innovation. This project aims to develop a predictive model for traffic signs based on image recognition techniques. By leveraging a dataset of various traffic signs, we can train a model that recognizes and classifies these signs accurately, enhancing driver safety and navigation.

## **Installation:**

Clone the repo:

```bash
$ git clone https://github.com/yourusername/Traffic-Sign-Prediction.git
```

Open the project's folder:

```bash
$ cd Traffic-Sign-Prediction
```

Install the dependencies:

```bash
$ pip install -r requirements.txt
```

Run the web application:

```bash
$ python manage.py migrate
$ python manage.py runserver
```

Navigate to the URL: [http://localhost:8000/](http://localhost:8000/)

*In case of any issues, please try on an Incognito Tab in your browser.*

## Technologies Used:

**Library:** OpenCV, Numpy, Pandas, TensorFlow/Keras

## Architecture of the System:

[Architecture flowchart](https://github.com/yourusername/Traffic-Sign-Prediction/blob/main/media/flowchart.png)

#### Local Binary Patterns Histogram

Local Binary Pattern (LBP) is an effective texture operator that labels the pixels of an image by thresholding the neighborhood of each pixel, resulting in a binary number. This method, combined with histograms, allows us to represent face images as simple data vectors, facilitating accurate recognition.

**Benefits:**

- LBPH is one of the simplest yet most effective face recognition algorithms.
- It requires minimal computational power and offers low time complexity.
- High accuracy in face recognition tasks.
- Supported by the OpenCV library.

## Traffic Sign Prediction Based on Images

### Dataset Collection

The dataset for this project comprises images of various traffic signs, ensuring that the data used for training the model is comprehensive and representative. The dataset is crucial for the model to learn the features of different traffic signs effectively.

### Training Algorithm

#### Convolutional Neural Networks (CNN)

Due to the advancements in computer vision and deep learning, CNNs are employed to achieve high accuracy in image classification tasks. The project utilizes structured data from the traffic signs dataset, employing CNNs to learn and predict the signs effectively.

The trained model achieves an impressive accuracy rate, making it suitable for real-time traffic sign recognition applications.

## Screenshots of the Web App:

<span align="left">
  <img width="400" height="300" src="https://github.com/Blessjr/Traffic-Sign-Prediction/blob/main/media/home.png">
</span>
<span align="right">
  <img width="400" height="300" src="https://github.com/Blessjr/Traffic-Sign-Prediction/blob/main/media/upload.png">
</span>
<span align="left">
  <img width="400" height="300" src="https://github.com/Blessjr/Traffic-Sign-Prediction/blob/main/media/login.png">
</span>
<span align="right">
  <img width="400" height="300" src="https://github.com/Blessjr/Traffic-Sign-Prediction/blob/main/media/prediction.png">
</span>

## Future Scope:

Future enhancements may include integrating a navigation system that can provide route guidance based on traffic sign predictions. Additionally, the system could expand to recognize a wider variety of traffic signs and include features for real-time updates based on traffic conditions.
