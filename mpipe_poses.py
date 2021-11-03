import cv2
import os

import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

train_path = "./trimmed/train"
vid = os.listdir(train_path)[0]
cap = cv2.VideoCapture(os.path.join(train_path, vid))


