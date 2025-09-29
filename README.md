# Emotion Recognition & AI Focus Coach

A real-time emotion recognition system that uses computer vision and machine learning to detect user focus/mood from facial cues, and provide voice reminders to help maintain attention.

Built with:

- OpenCV + MediaPipe (FaceMesh via cvzone) for face landmark detection
- Custom-trained ML model for emotion classification
- Temporal smoothing for stable predictions
- pyttsx3 for speech feedback (AI coach)

## Features
- Live camera feed with face mesh overlay
- Real-time emotion detection
- Stabilized predictions using rolling history
- AI speech agent that gives supportive reminders when certain emotions are detected

## Screenshot
<img width="419" height="272" alt="image" src="https://github.com/user-attachments/assets/baf511ac-93ac-4dc7-9b35-184bb2f2c108" />

## Installation

1) Clone:
- git clone https://github.com/yourusername/emotion-recognition.git
- cd emotion-recognition

2) Install dependencies:
- pip install -r requirements.txt

3) Run:
- python pal.py

## Credits

This project builds on https://github.com/Tech-Watt/Emotion-Detection for the MediaPipe FaceMesh setup, and extends it with:
- A trained behavior model
- Stability logic
- A speech-based AI focus coach
