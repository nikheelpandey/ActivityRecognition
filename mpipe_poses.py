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



DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

    return img

with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:

    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            image = resize(frame)
            # cv2.imshow('Frame',frame)

            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print nose landmark.
            image_hight, image_width, _ = image.shape
        
            if not results.pose_landmarks:
                continue
        
            print(
                f'Nose coordinates: ('
                f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
                f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_hight})'
            )

            print(f'results.pose_landmarks \n{results.pose_landmarks}')
            print(f'results.POSE_CONNECTIONS \n{results.POSE_CONNECTIONS}')

            # # Draw pose landmarks.
            # print(f'Pose landmarks of {name}:')
            # annotated_image = image.copy()
            # mp_drawing.draw_landmarks(
            #     annotated_image,
            #     results.pose_landmarks,
            #     mp_pose.POSE_CONNECTIONS,
            #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # resize_and_show(annotated_image)