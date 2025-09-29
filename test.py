import cvzone
import pickle
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
FMD = FaceMeshDetector()

with open('model.pkl','rb') as f:
   Behaviour_model = pickle.load(f)

# Temporal smoothing variables
STABILITY_DURATION = 1.0  # seconds
FPS_ESTIMATE = 30  # approximate fps
REQUIRED_FRAMES = int(FPS_ESTIMATE * STABILITY_DURATION)  # frames needed for stability

emotion_history = []
displayed_emotion = "Unknown"
frame_count = 0

# taking video frame by frame
while cap.isOpened():
    rt, frame = cap.read()
    if not rt:
        break
        
    frame = cv2.resize(frame, (720, 480))
    frame_count += 1
    
    real_frame = frame.copy()
    img, faces = FMD.findFaceMesh(frame)
    
    cvzone.putTextRect(frame, ('Mood'), (10, 80))
    
    current_prediction = None
    
    if faces:
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
            
            # Debug: print current prediction and history length
            if frame_count % 30 == 0:  # print every 30 frames
                print(f"Current: {current_prediction}, History length: {len(emotion_history)}")
                if len(emotion_history) > 0:
                    # Count emotions in history
                    emotion_counts = {}
                    for emotion in emotion_history:
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    print(f"Emotion counts: {emotion_counts}")
            
        except Exception as e:
            print(f"Prediction error: {e}")
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
    
    # Display the emotion
    cvzone.putTextRect(frame, str(displayed_emotion), (250, 80))
    print(f"Displayed Emotion: {displayed_emotion}")
    
    # Show debug info
    debug_info = f"History: {len(emotion_history)}/{REQUIRED_FRAMES}"
    cvzone.putTextRect(frame, debug_info, (10, 120), scale=0.7)
    
    if current_prediction:
        current_info = f"Current: {current_prediction}"
        cvzone.putTextRect(frame, current_info, (10, 150), scale=0.7)
    
    all_frames = cvzone.stackImages([real_frame, frame], 2, 0.70)
    cv2.imshow('frame', all_frames)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()