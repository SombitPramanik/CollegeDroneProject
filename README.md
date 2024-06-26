# Face Recognition System with OpenCV

This documentation provides a detailed guide on creating a face recognition system using OpenCV. The approach involves storing data in a folder, processing this data into a pickle model, and then using this model to recognize faces in real-time. The system is designed to run on a Raspberry Pi 4B with 8GB of RAM, targeting a frame rate of 12 FPS.

## Prerequisites

### Hardware
- Raspberry Pi 4B with 8GB of RAM
- Camera (compatible with Raspberry Pi)
- Power supply for Raspberry Pi

### Software
- Raspbian OS (or any compatible OS for Raspberry Pi)
- Python 3.x
- OpenCV
- dlib
- face_recognition
- numpy
- pickle

## Steps to Create the Face Recognition System

### 1. Setting Up the Environment

First, ensure that your Raspberry Pi is up to date:

```bash
sudo apt update
sudo apt upgrade
```

Next, install the necessary libraries:

```bash
sudo apt install python3 python3-pip
pip3 install opencv-python opencv-python-headless dlib face_recognition numpy
```

### 2. Preparing the Dataset

Store all your images in a folder, for example `dataset/`. Each image should be named according to the person in the picture, e.g., `person1.jpg`, `person2.jpg`, etc.

### 3. Extracting Data and Creating the Pickle Model

Create a script to process the images and generate the pickle model:

```python
import os
import face_recognition
import pickle

# Path to the dataset
dataset_path = 'dataset/'

# Initialize lists
known_encodings = []
known_names = []

# Loop over the images in the dataset
for filename in os.listdir(dataset_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        image = face_recognition.load_image_file(os.path.join(dataset_path, filename))

        # Get the face encoding
        encodings = face_recognition.face_encodings(image)

        if encodings:
            # Save the encoding and the name
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])

# Save the encodings and names to a pickle file
with open('encodings.pkl', 'wb') as f:
    pickle.dump((known_encodings, known_names), f)
```

### 4. Real-time Face Recognition

Create a script to perform real-time face recognition:

```python
import cv2
import face_recognition
import pickle
import numpy as np

# Load the known faces and names
with open('encodings.pkl', 'rb') as f:
    known_encodings, known_names = pickle.load(f)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Target frames per second
fps = 12

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all face locations and face encodings in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if name == "Unknown":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Sleep to target the FPS
    cv2.waitKey(int(1000 / fps))

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
```

### 5. Running the System

To start the face recognition system, simply run the real-time face recognition script:

```bash
python3 face_recognition_realtime.py
```

### Conclusion

This guide provides a comprehensive overview of setting up a face recognition system using OpenCV on a Raspberry Pi 4B. The system loads pre-processed face data from a pickle model and performs real-time face recognition, marking recognized faces with green rectangles and unknown faces with red rectangles. The target frame rate is 12 FPS, ensuring smooth performance on the specified hardware.
