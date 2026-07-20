Real-Time Face Mask Detection using YOLO11

## Overview

This project is a **real-time face mask detection system** developed using **Ultralytics YOLO11**, **OpenCV**, and **Python**. The application detects whether a person is wearing a face mask through a live webcam feed. If a person is detected **without a face mask**, the system automatically captures an image and sends an **email alert** with the detected image attached.

To ensure smooth real-time performance, email notifications are sent using **Python multithreading**, preventing the webcam stream from freezing during email transmission.

## 🚀 Features

- Real-time webcam monitoring
- Detects **Mask** and **No Mask**
- Automatic email alerts with captured image
- Saves violation images locally
- Uses multithreading for uninterrupted detection
- Custom-trained YOLO11 model
- Lightweight and suitable for real-time monitoring

## Technologies Used

- Python
- Ultralytics YOLO11
- OpenCV
- PyTorch
- SMTP
- Python Threading

## Project Structure

Face-Mask-Detection/
│
├── dataset/                     # Training dataset
├── runs/
│   └── detect/
│       └── train-3/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           ├── results.png
│           ├── confusion_matrix.png
│           ├── BoxPR_curve.png
│           ├── BoxP_curve.png
│           ├── BoxR_curve.png
│           └── ...
│
├── alerts/                     
├── webcam.py                   
├── train.py                     
├── requirements.txt
├── README.md
└── .gitignore
 
 ## How it works:

The application will:

1. Open the webcam.
2. Detect people wearing and not wearing face masks.
3. Display bounding boxes with confidence scores.
4. Capture an image when a **No Mask** person is detected.
5. Send an email alert with the captured image with the timing.
6. Save the captured image inside the **alerts** folder.

##  Email Alert

Whenever a **No Mask** person is detected, the system automatically:

- Captures the current frame
- Saves the image
- Sends an email notification
- Attaches the captured image
- Prevents repeated alerts using a time interval

---

##  Model Information

- Model: **YOLO11n**
- Framework: **Ultralytics YOLO**
- Epochs: 6
- Detection Classes:
  - Mask
  - No Mask

The model was trained using a custom face mask dataset and exported as **best.pt**, which is used during inference.

---

## 📊 Training Results

 Precision : 68.75%
 Recall    : 66.15%
 mAP@0.5   : 70.74%


## Sample Output

A demonstration of the real-time face mask detection system is available on my LinkedIn profile.

LinkedIn account: https://www.linkedin.com/in/darunsuresh

## Detection

-  Detects people wearing face masks
-  Detects people without face masks
-  Displays bounding boxes with confidence scores

### Alert

When a **No Mask** person is detected:

- Image is captured
- Image is saved locally
- Email notification is sent automatically

---

## Future Improvements

- Multiple camera support
- Face recognition with mask detection
- Live analytics dashboard

## Learning Outcomes

Through this project, I gained practical experience in:

- Computer Vision
- Object Detection
- YOLO11 Training
- OpenCV
- PyTorch
- Email Automation
- Python Multithreading
- Real-time AI Applications



## Author

**Darun**

Computer Science Engineering Student

## Suggestions
 
 If you have any suggestions, feedback, or ideas to improve this project, feel free to reach out via email. Your feedback is greatly appreciated.
 Email ID: darunsuresh07@gmail.com
 

Interested in:
- Artificial Intelligence
- Machine Learning
- Computer Vision
- Deep Learning
