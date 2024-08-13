# Hardhat Detection Model
This repository contains a Convolutional Neural Network (CNN) model built using TensorFlow and Keras, designed to classify images into two categories: "hardhat" and "nohardhat." The model is trained on a dataset of images and can be used to predict whether a person in an image is wearing a hard hat or not.

## Table of Contents
Overview  
Requirements  
Setup  
Usage  
Model Architecture  
Training  
Evaluation  
Testing  
Results  
Contributing  
License  
# Overview
## The code in this repository performs the following tasks:

Data Preprocessing: It removes corrupt images from the dataset.
Data Loading: It loads the images from the dataset and prepares them for training.
Model Creation: It defines and compiles a CNN model.
Training: It trains the model on the dataset.
Evaluation: It evaluates the model's performance on test data.
Testing: It tests the model on a sample image to predict whether a hardhat is present.
# Requirements
To run this project, you need to just install requrements.txt

# Structure 
Ensure you have a dataset organized in the following directory structure:
```
logs/  
classes/  
├── hardhat/  
│   ├── image1.jpg  
│   ├── image2.jpg  
│   └── ...  
└── nohardhat/  
    ├── image1.jpg  
    ├── image2.jpg  
    └── ...
```
Place the dataset in the classes directory.

# Usage
This script will:

Preprocess the dataset by removing corrupt images.
Load the dataset and prepare it for training.
Train the CNN model.
Evaluate the model's performance.
Test the model on a sample image.
Model Architecture
The model is a Sequential CNN with the following layers:

Input Layer: Accepts images of shape (256, 256, 3).
Convolutional Layers: 3 convolutional layers with ReLU activation and max pooling.
Flatten Layer: Flattens the 2D matrices into vectors.
Dense Layers: 2 fully connected layers, the last of which uses a sigmoid activation function for binary classification.
Training
The model is trained using the Adam optimizer and binary cross-entropy loss for 20 epochs. The training and validation losses and accuracies are plotted for performance analysis.

# Evaluation
The model is evaluated using precision, recall, and accuracy metrics on the test dataset.

# Testing
You can test the model on a new image (e.g., nohardhattest.jpg) to predict whether a hardhat is present  by running following cell

```
img = cv2.imread('nohardhattest.jpg')
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize/255, 0))

if yhat > 0.5:
    print('Predicted class is nohardhat')
else:
    print('Predicted class is hardhat')
```

# Results
The results of the evaluation will print the model's precision, recall, and accuracy. The model's performance will also be plotted using Matplotlib.

# Known limitations
This is a simple architecture that may not pickup specifically on hardhats as a feature

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any bugs or suggestions.

# License
This project is licensed under the MIT License
