# FaceRecog - Face Recognition System

FaceRecog is a Python-based face recognition system built with OpenCV and featuring a user-friendly GUI using Tkinter. It uses a DNN-based face detector (SSD with ResNet) for detection and LBPH for recognition.

## Features
- Real-time Face Detection: Uses DNN (SSD with ResNet) for accurate face detection.
- Face Recognition: Identifies known faces with confidence scores using LBPH.
- GUI Interface: Buttons for camera control, training, and adding new faces.
- Training System: Captures and stores face samples for training.
- Persistent Model: Saves trained models and labels for reuse.

## Prerequisites
- Python 3.x
- Required libraries:
  - `opencv-python`
  - `opencv-contrib-python` (for LBPH)
  - `pillow`
  - `numpy`
- DNN model files:
  - `deploy.prototxt`
  - `res10_300x300_ssd_iter_140000.caffemodel`

## Installation
1. Clone or download this repository:
   ```bash
   git clone <repository-url>
   cd FaceRecog
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python opencv-contrib-python pillow numpy
   ```
3. Download DNN model files:
   - `deploy.prototxt`: [Link](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)
   - `res10_300x300_ssd_iter_140000.caffemodel`: [Link](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)
   Place these in the project directory.
4. Ensure a working webcam is connected.

## Usage
1. Run the application:
   ```bash
   python facerecog.py
   ```
2. Use the GUI:
   - **Start Camera**: Begins real-time DNN face detection and recognition.
   - **Stop Camera**: Stops the video feed.
   - **Add New Face**: Prompts for a name and captures 30 face samples.
   - **Train Model**: Trains the LBPH recognition system.

3. Training Data:
   - Stored in `training_data/<person_name>/`.
   - Images saved in grayscale.

4. Recognition:
   - Known faces show names and confidence scores.
   - Unknown faces labeled "Unknown."

## File Structure
```
FaceRecog/
├── facerecog.py              # Main script
├── deploy.prototxt           # DNN model architecture
├── res10_300x300_ssd_iter_140000.caffemodel  # DNN pretrained weights
├── training_data/           # Face samples (auto-created)
├── trained_model.yml        # Trained LBPH model (auto-generated)
├── labels.pickle            # Label mappings (auto-generated)
└── README.md                # This file
```

## How It Works
- **Detection**: DNN (SSD with ResNet) detects faces with high accuracy.
- **Recognition**: LBPHFaceRecognizer identifies faces from trained data.
- **Storage**: Grayscale images stored for training.
- **UI**: Tkinter interface for user interaction.

## Notes
- Requires good lighting and clear face visibility.
- Captures 30 samples per new face.
- Training required after adding new faces.
- Model persists via `trained_model.yml` and `labels.pickle`.

## Potential Improvements
- Add face alignment for better LBPH accuracy.
- Integrate DeepFace or other advanced recognition.
- Add database support.
- Enhance UI with delete/list features.

## Troubleshooting
- **No camera feed**: Check webcam connection/permissions.
- **Model files missing**: Download `deploy.prototxt` and `.caffemodel`.
- **Training error**: Verify `training_data` contains valid images.

## License
MIT License

## Acknowledgments
- Built with OpenCV, Tkinter, and Python.
- DNN model from OpenCV contributors.

---
Developed by [Mit Patel] | March 2025
```

### Key Changes
1. **Detection**: Replaced Haar Cascade with DNN (SSD with ResNet) for better accuracy.
2. **Recognition**: Kept LBPH as it's lightweight and suitable for local recognition tasks.
3. **Dependencies**: Added requirement for DNN model files.
4. **Performance**: DNN provides more robust detection than Haar, though slightly slower.
5. **UI**: Maintained the same functionality with updated backend.

### Setup Instructions
- Download the DNN model files as described in the README.
- Place them in the same directory as `facerecog.py`.
- Install dependencies and run the script.
