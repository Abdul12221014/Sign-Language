import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pyttsx3
import cv2

# Load Dataset
train_data_path = '/Users/abdulkadir/Downloads/Sign Language/sign_mnist_train/sign_mnist_train.csv'
test_data_path = '/Users/abdulkadir/Downloads/Sign Language/sign_mnist_test/sign_mnist_test.csv'

# Load train and test CSV data
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Preprocess the data
X_train = train_data.drop('label', axis=1).values / 255.0  # Normalize pixel values
y_train = to_categorical(train_data['label'].values, 26)  # Convert labels to one-hot encoding
X_test = test_data.drop('label', axis=1).values / 255.0  # Normalize pixel values
y_test = to_categorical(test_data['label'].values, 26)  # Convert labels to one-hot encoding

# Reshape the data to match input dimensions (28x28 pixels, 1 channel)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')  # 26 classes (A-Z)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model after training
model.save('sign_language_model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()

# Function to convert text to speech
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Example function to predict and speak the recognized sign
def predict_and_speak(image_path):
    # Read the image (assuming image is 28x28 pixels)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))  # Resize to 28x28 if necessary
    img = img / 255.0  # Normalize pixel values
    img = img.reshape(1, 28, 28, 1)  # Reshape for the model input

    # Predict the character
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)

    # Convert label to letter
    predicted_char = chr(predicted_label + 65)  # Convert to A-Z
    print(f"Predicted Letter: {predicted_char}")

    # Speak the predicted letter
    speak_text(predicted_char)

# Test the prediction and speech (example with one image)
predict_and_speak('/Users/abdulkadir/Downloads/Sign Language/amer_sign2.png')
