import cv2
import os
import math
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
# mp_drawing_styles = mp.solutions.drawing_styles

train_path = "trimmed/train/"
classes = os.listdir(train_path)


DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)


from tqdm import tqdm

for activity in classes:

    file_path = os.path.join(train_path,str(activity))
    
    for idx, vid in tqdm(enumerate( os.listdir(file_path))):
        video_path = os.path.join(file_path,vid)
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        # print(cap.read())

        while(cap.isOpened()):
        # Capture frame-by-frame
            ret, frame = cap.read()


            if ret == True:

                image = cv2.resize(frame, (DESIRED_WIDTH, DESIRED_WIDTH))
                # cv2.imshow('Frame',frame)

                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Print nose landmark.
                image_hight, image_width, _ = image.shape
            
                if not results.pose_landmarks:
                    continue
                
                

                # print(f'results.pose_landmarks \n{results.pose_landmarks}')
