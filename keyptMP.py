#USING MEDIAPIPE
import math
import time
from glob import glob
from pathlib import Path
import cv2
import csv
import pandas as pd
from pydantic import BaseModel
import mediapipe as mp
import os
os.makedirs('dataset', exist_ok=True)

scale_factor = 0.5
def resize_image(image, scale_factor):
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    return cv2.resize(image, (new_width, new_height))

# Keypoint constants
class GetKeypoint(BaseModel):
    NOSE: int = 0
    LEFT_EYE: int = 1
    RIGHT_EYE: int = 2
    LEFT_EAR: int = 3
    RIGHT_EAR: int = 4
    LEFT_SHOULDER: int = 11
    RIGHT_SHOULDER: int = 12
    LEFT_ELBOW: int = 13
    RIGHT_ELBOW: int = 14
    LEFT_WRIST: int = 15
    RIGHT_WRIST : int = 16 
    LEFT_HIP: int = 23
    RIGHT_HIP: int = 24
    LEFT_KNEE: int = 25
    RIGHT_KNEE: int = 26
    LEFT_ANKLE: int = 27
    RIGHT_ANKLE: int = 28

get_keypoint = GetKeypoint()
os.makedirs('dataset', exist_ok=True)

for source_path in glob('Videos/*'):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
    mp_drawing = mp.solutions.drawing_utils

    for source_path in glob('Videos/*'):
        name = Path(source_path).stem
        cap = cv2.VideoCapture(source_path)

        prev_angle = None
        prev_angular_velocity = None
        linear_acceleration = None
        prev_time = 0
        data_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            try:
                frame = resize_image(frame, scale_factor)
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    def get_coords(index):
                        return (landmarks[index].x, landmarks[index].y)

                    right_ankle = get_coords(get_keypoint.RIGHT_ANKLE)
                    right_knee = get_coords(get_keypoint.RIGHT_KNEE)
                    right_hip = get_coords(get_keypoint.RIGHT_HIP)

                    # Vectors
                    vector1 = (right_knee[0] - right_ankle[0], right_knee[1] - right_ankle[1])
                    vector2 = (right_hip[0] - right_ankle[0], right_hip[1] - right_ankle[1])

                    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
                    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
                    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
                    angle_rad = math.acos(dot_product / (magnitude1 * magnitude2 + 1e-6))  # add epsilon to avoid zero-div
                    angle_deg = math.degrees(angle_rad)

                    time_diff = current_time - prev_time

                    angular_velocity = None
                    if prev_angle is not None and time_diff > 0:
                        angular_velocity = (angle_deg - prev_angle) / time_diff
                        print(f'Angular Velocity: {angular_velocity:.2f} deg/s')

                    if prev_angular_velocity is not None and angular_velocity is not None:
                        linear_acceleration = (angular_velocity - prev_angular_velocity)

                    prev_angle = angle_deg
                    prev_angular_velocity = angular_velocity
                    prev_time = current_time

                    data_list.append({
                        'time': current_time,
                        'linear_acceleration': linear_acceleration,
                        'angular_velocity': angular_velocity
                    })
            except Exception as e:
                print(e)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        df = pd.DataFrame(data_list)
        df.to_csv(f'dataset/{name}.csv')

    pose.close()
    cv2.destroyAllWindows()
