Face Mask Detection Using MobileNetV2
A real-time Face Mask Detection System built using TensorFlow, Keras, and OpenCV. The model classifies faces as:
With Mask 😷
Without Mask 🙍‍♂️
This project was developed as part of an assignment to demonstrate skills in AI, Deep Learning, and Computer Vision.

📁 Project Structure
face-mask-detection/
│
├── train.py # Model training script
├── detect.py # Real-time mask detection with webcam
├── requirements.txt # Required Python packages
├── model/ # Saved trained model (.h5)
├── dataset/
│ ├── train/
│ │ ├── with_mask/
│ │ └── without_mask/
│ └── val/
│ ├── with_mask/
│ └── without_mask/
└── screenshots/ # Output images

🚀 Features
Real-time face detection using OpenCV Haar Cascade
Mask classification using MobileNetV2
High accuracy with a large dataset
Works on CPU and GPU
Lightweight model suitable for deployment

🔧 Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/face-mask-detection-using-mobilenetv2.git
cd face-mask-detection-using-mobilenetv2
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Add dataset
Create this structure:
dataset/train/with_mask/
dataset/train/without_mask/
dataset/val/with_mask/
dataset/val/without_mask/

🧠 Training the Model
Run:
python train.py
The model will be saved automatically to:
model/face_mask_mobilenetv2.h5

🎥 Real-Time Detection
Connect a webcam and run:
python detect.py
Green Box → With Mask
Red Box → Without Mask

📌 Technologies Used
Python
TensorFlow / Keras
OpenCV
MobileNetV2
NumPy
