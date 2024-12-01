import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model (change the path to where your model is stored)
model = load_model('/Users/abdulkadir/Downloads/Sign Language/sign_language_model.h5')

# Initialize the camera (0 is usually the default camera, 1 or 2 if you have multiple cameras)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If the frame was not captured properly, exit
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Debug: Check the frame size
    print(f"Original frame size: {frame.shape}")

    # Preprocess the captured frame
    # Uncomment the next line if your model expects RGB images
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to grayscale if the model was trained with grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray, (28, 28))  # Resize the frame to 28x28
    resized_frame = resized_frame / 255.0  # Normalize pixel values
    resized_frame = np.expand_dims(resized_frame, axis=-1)  # Add channel dimension (1 for grayscale)
    resized_frame = np.expand_dims(resized_frame, axis=0)  # Add batch dimension

    # Debug: Check the shape of resized frame
    print(f"Resized frame shape: {resized_frame.shape}")

    # Predict the gesture using the model
    prediction = model.predict(resized_frame)

    # Get the predicted class (0-25 for A-Z)
    predicted_class = np.argmax(prediction)

    # Create a label with the predicted sign language letter
    label = chr(predicted_class + 65)  # Convert to A-Z based on index

    # Display the result on the frame
    cv2.putText(frame, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the video feed with prediction
    cv2.imshow('Sign Language Recognition', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
