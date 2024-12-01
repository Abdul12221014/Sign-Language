# Sign Language Recognition

This project uses a deep learning model to recognize American Sign Language (ASL) gestures from images. The model has been trained on the Sign MNIST dataset and is capable of predicting ASL letters based on hand gesture images.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model](#model)
4. [Usage](#usage)
5. [Requirements](#requirements)
6. [License](#license)

## Introduction

This project aims to build a system that can recognize American Sign Language gestures and predict the corresponding letter (A-Z). The model was trained using the Sign MNIST dataset, which consists of images representing different hand gestures for the alphabet. The system captures real-time video feed from the webcam, processes the captured frames, and predicts the sign language letter using the trained model.

## Dataset

The dataset used for training is the **Sign MNIST** dataset, which consists of images of hand gestures corresponding to the ASL alphabet (A-Z). Each image is a 28x28 grayscale image of the hand in a specific position representing a letter. The dataset has been used for training and evaluation of the model.

## Model

The model used in this project is a convolutional neural network (CNN) trained on the Sign MNIST dataset. The model architecture is designed to classify the 26 possible ASL hand gestures.

Model file: `sign_language_model.h5`

## Usage

### Running the Script

To use the sign language recognition system, follow these steps:

1. Clone this repository:

    ```bash
    git clone https://github.com/Abdul12221014/Sign-Language.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Sign-Language
    ```

3. Ensure you have the required libraries installed (listed below).

4. Run the script to start the real-time sign language recognition system:

    ```bash
    python3 Testing.py
    ```

The webcam will activate, and the model will start predicting the ASL gestures. The predicted letter will be displayed on the video feed.

### Model Prediction Example

The model predicts letters based on the hand gesture captured from the webcam. For example:

- When the hand gesture representing the letter "A" is captured, the system will display "Prediction: A".

## Requirements

To run the project, you need the following dependencies:

- Python 3.x
- OpenCV
- TensorFlow
- NumPy

You can install the required packages using pip:

```bash
pip install opencv-python tensorflow numpy
