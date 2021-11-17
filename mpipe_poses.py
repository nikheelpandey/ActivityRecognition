import cv2
import os
import math
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
# mp_drawing_styles = mp.solutions.drawing_styles

train_path = "trimmed/val/"
classes = os.listdir(train_path)


data = {}
data['video_key_point'] = []

import json



DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)


from tqdm import tqdm

for idx_, activity in enumerate(classes):

    file_path = os.path.join(train_path,str(activity))
    
    for idx, vid in tqdm(enumerate( os.listdir(file_path))):

        if str(vid)+'_data_val.json' is os.listdir("./"):

            continue

        # try:
        video_path = os.path.join(file_path,vid)
        

        landmarks_ls = [] 

        # print(video_path)
        cap = cv2.VideoCapture(video_path)
        # print(cap.read())

        while(cap.isOpened()):
        # Capture frame-by-frame
            ret, frame = cap.read()


            if ret == True:

                image = cv2.resize(frame, (DESIRED_WIDTH, DESIRED_WIDTH))
                # cv2.imshow('Frame',frame)
                try:
                    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                except:
                    print("Corrupt frame")
                    continue
                # Print nose landmark.
                image_hight, image_width, _ = image.shape
            
                if not results.pose_landmarks:
                    continue
                
                # print(len(results.pose_landmarks.landmark))
                
                frame_landmark_ls = []
                for landmarks in results.pose_landmarks.landmark:
                    frame_landmark_ls.append([int(landmarks.x*480),int(landmarks.y*480),int(landmarks.z*480)])
                
                landmarks_ls.append(frame_landmark_ls)

                # print(f'results.pose_landmarks \n{results.pose_landmarks}')
            else:
                
                vid_data = {}
                vid_data["filename"] = {"name":vid,
                                        'class':idx_ ,
                                        'keypoints':landmarks_ls} 
                


                data["video_key_point"].append(vid_data)
                
                break


        with open(str(vid)+'_data_val.json', 'w') as f:
            json.dump(data, f)
        
        