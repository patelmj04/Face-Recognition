# Face Recog - Face Recognition System

FaceRecog is a Python-based face recognition system built with OpenCV and featuring a user-friendly GUI using Tkinter. It allows users to detect faces in real-time, train the system with new faces, and recognize known individuals using the LBPH (Local Binary Patterns Histograms) algorithm.

## Features
- **Real-time Face Detection**: Detects faces using Haar Cascade Classifier.
- **Face Recognition**: Identifies known faces with confidence scores.
- **GUI Interface**: Includes buttons for starting/stopping the camera, training the model, and adding new faces.
- **Training System**: Captures and stores face samples for training.
- **Persistent Model**: Saves trained models and labels for reuse.

## Prerequisites
- Python 3.x
- Required libraries:
  - `opencv-python`
  - `opencv-contrib-python`
  - `pillow`
  - `numpy`

## Installation
1. Clone or download this repository:
   ```bash
   git clone <repository-url>
   cd FaceRecog
   Install the required dependencies:
bash

Collapse

Wrap

Copy
pip install opencv-python opencv-contrib-python pillow numpy
Ensure you have a working webcam connected to your system.
Usage
Run the application:
bash

Collapse

Wrap

Copy
python facerecog.py
Use the GUI:
Start Camera: Begins real-time face detection and recognition.
Stop Camera: Stops the video feed.
Add New Face: Prompts for a name and captures 30 face samples.
Train Model: Trains the recognition system with collected data.
Training Data:
Face samples are stored in the training_data directory.
Each person’s images are saved in a subdirectory named after them.
Recognition:
Known faces display with names and confidence scores.
Unknown faces are labeled as "Unknown."
File Structure
text

Collapse

Wrap

Copy
FaceRecog/
├── facerecog.py         # Main application script
├── training_data/       # Directory for storing face samples (auto-created)
├── trained_model.yml    # Trained recognition model (auto-generated)
├── labels.pickle        # Label mappings (auto-generated)
└── README.md            # This file
How It Works
Detection: Uses OpenCV’s Haar Cascade Classifier for face detection.
Recognition: Employs LBPHFaceRecognizer for identifying faces.
Storage: Saves images in grayscale for training and recognition.
UI: Tkinter provides a simple interface for user interaction.
Notes
Ensure good lighting and clear face visibility for best results.
The system captures 30 samples per new face by default.
Training must be performed after adding new faces for recognition to work.
The model persists between sessions via trained_model.yml and labels.pickle.
Potential Improvements
Add face alignment for better accuracy.
Integrate more advanced recognition algorithms (e.g., DeepFace).
Add database support for managing users.
Enhance UI with additional features (e.g., delete person, list faces).
Troubleshooting
No camera feed: Check webcam connection and permissions.
Training error: Ensure the training_data folder contains valid images.
Dependencies: Verify all libraries are installed correctly.
License
This project is open-source and available under the MIT License.

Acknowledgments
Built with OpenCV, Tkinter, and Python.
Thanks to the open-source community for invaluable resources and tools.
Developed by [Your Name] | March 2025

text

Collapse

Wrap

Copy

### Customization Notes:
- Replace `<repository-url>` with your actual repo URL if you’re hosting it online.
- Add your name in the "Developed by" section if you want credit!
- Swap "FaceRecog" with "FaceLook" or another name if you decide otherwise.
- Adjust the date or any specifics based on your needs.

This README provides a clear overview, setup instructions, and usage guide for anyone picking