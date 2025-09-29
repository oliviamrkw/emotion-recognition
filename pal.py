# test_with_speech.py
import cvzone
import pickle
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import numpy as np
import time
import threading
import random

# Optional TTS: pyttsx3
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

def speak(text):
    """Simple wrapper to speak text. Creates engine per-call to avoid thread issues."""
    if pyttsx3 is None:
        print("(TTS not installed) ->", text)
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("TTS error:", e)
        # fallback: just print
        print(text)

def speak_in_background(text):
    t = threading.Thread(target=speak, args=(text,), daemon=True)
    t.start()


# ---------------- Load camera, face mesh, model ----------------
cap = cv2.VideoCapture(0)
FMD = FaceMeshDetector()

try:
    with open('model.pkl','rb') as f:
       Behaviour_model = pickle.load(f)
    print("✅ model.pkl loaded")
except Exception as e:
    print("❌ Error loading model.pkl:", e)
    Behaviour_model = None

# Temporal smoothing variables (same as original)
STABILITY_DURATION = 1.0  # seconds
FPS_ESTIMATE = 30  # approximate fps
REQUIRED_FRAMES = int(FPS_ESTIMATE * STABILITY_DURATION)  # frames needed for stability

emotion_history = []
displayed_emotion = "Unknown"
frame_count = 0

# ---------------- Emotion -> Speech Responses ----------------
emotion_responses = {
    'distracted': [
        "Hey, I noticed you seem a bit distracted. Let's get back to focus!",
        "Time to refocus! You've got this."
    ],
    'tired': [
        "You look tired. Maybe take a short break or have some water?",
        "Consider a quick stretch to refresh yourself."
    ],
    'confused': [
        "You look confused, do you need me to explain this topic further?",
        "Take a small break and tackle one thing at a time — you've got this."
    ],
    'happy': [
        "Great to see you happy!",
        "You're in a good mood — keep it up!"
    ]
}

# per-emotion cooldowns (seconds)
last_intervention_time = {}
INTERVENTION_COOLDOWN = 10  # seconds
speech_enabled = True

def should_intervene(emotion):
    if not emotion or emotion == "Unknown":
        return False
    now = time.time()
    last = last_intervention_time.get(emotion, 0)
    if now - last >= INTERVENTION_COOLDOWN:
        last_intervention_time[emotion] = now
        return True
    return False

# ---------------- Main loop ----------------
print("Starting camera. Press Q to quit, M to toggle speech.")
while cap.isOpened():
    rt, frame = cap.read()
    if not rt:
        # avoid tight spin if camera read fails
        time.sleep(0.05)
        continue

    frame = cv2.resize(frame, (720, 480))
    frame_count += 1

    real_frame = frame.copy()
    img, faces = FMD.findFaceMesh(frame)

    cvzone.putTextRect(frame, ('Mood'), (10, 80))

    current_prediction = None

    if faces and Behaviour_model is not None:
        face = faces[0]
        face_data = list(np.array(face).flatten())

        try:
            # Get prediction from model
            result = Behaviour_model.predict([face_data])
            current_prediction = result[0]

            # Add to history
            emotion_history.append(current_prediction)

            # Keep only recent history (last REQUIRED_FRAMES frames)
            if len(emotion_history) > REQUIRED_FRAMES:
                emotion_history.pop(0)

            # Debug logging occasionally
            if frame_count % 60 == 0:
                print(f"[debug] Current: {current_prediction}, History: {len(emotion_history)}")

        except Exception as e:
            # prediction failed for this frame
            print("Prediction error:", e)
            current_prediction = None

    # Update displayed emotion if we have enough stable history
    if len(emotion_history) >= min(10, REQUIRED_FRAMES // 3):  # Start showing after 10 frames or 1/3 of required
        # Count occurrences of each emotion
        emotion_counts = {}
        for emotion in emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        # Get most common emotion
        most_common_emotion = max(emotion_counts, key=emotion_counts.get)
        most_common_count = emotion_counts[most_common_emotion]

        # If we have enough frames and the most common emotion is dominant enough
        if len(emotion_history) >= REQUIRED_FRAMES:
            # Need 60% consistency for full buffer
            if most_common_count / len(emotion_history) >= 0.6:
                displayed_emotion = most_common_emotion
        else:
            # For partial buffer, need 70% consistency
            if most_common_count / len(emotion_history) >= 0.7:
                displayed_emotion = most_common_emotion

    # ---------------- Trigger speech if appropriate ----------------
    if speech_enabled and displayed_emotion in emotion_responses:
        if should_intervene(displayed_emotion):
            # pick a message and speak in background
            message = random.choice(emotion_responses[displayed_emotion])
            print("Coach:", message)
            speak_in_background(message)

    # Display the emotion on the frame
    cvzone.putTextRect(frame, str(displayed_emotion), (250, 80))

    # Show debug info
    debug_info = f"History: {len(emotion_history)}/{REQUIRED_FRAMES}"
    cvzone.putTextRect(frame, debug_info, (10, 120), scale=0.7)

    if current_prediction:
        current_info = f"Current: {current_prediction}"
        cvzone.putTextRect(frame, current_info, (10, 150), scale=0.7)

    # Indicate TTS state on screen
    cvzone.putTextRect(frame, f"Speech: {'ON' if speech_enabled else 'OFF'} (M toggle)", (10, 420), scale=0.6)

    all_frames = cvzone.stackImages([real_frame, frame], 2, 0.70)
    cv2.imshow('Emotion + Speech', all_frames)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break
    elif key in (ord('m'), ord('M')):
        speech_enabled = not speech_enabled
        print("Speech toggled:", speech_enabled)

cap.release()
cv2.destroyAllWindows()
