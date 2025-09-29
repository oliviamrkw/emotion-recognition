import pickle
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import numpy as np
import time
from collections import deque


cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture(0)
FMD = FaceMeshDetector()



with open('model.pkl','rb') as f:
   Behaviour_model = pickle.load(f)


# taking video frame by frame
while cap.isOpened():
    rt,frame = cap.read()
