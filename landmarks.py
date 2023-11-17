#landmarks.py
import os
import mediapipe as mp
import cv2
import numpy as np
from tkinter import filedialog
from tkinter import Tk
import csv

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to draw landmarks on image
def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    if detection_result.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            detection_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    return annotated_image

# Function to create a Pose object with higher confidence thresholds
def create_pose_object():
    return mp_pose.Pose(
        min_detection_confidence=0.5,  # Adjusted for better detection confidence
        min_tracking_confidence=0.5    # Adjusted for better tracking confidence
    )

# Function to save landmarks to a CSV file
def save_landmarks_to_csv(image_file_path, pose_landmarks, label):
    file_name = os.path.basename(image_file_path)
    data_rows = []
    headers = ['file_name', 'label', 'landmark_id', 'x', 'y', 'z']
    file_exists = os.path.isfile('landmarks.csv')
    for idx, landmark in enumerate(pose_landmarks.landmark):
        landmark_coords = [landmark.x, landmark.y, landmark.z]
        data_rows.append([file_name, label, idx] + landmark_coords)  # Include the label here
    
    with open('landmarks.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)  
        writer.writerows(data_rows)

# Function to process an image file
def process_image_file(image_file_path, pose, label):
    image = cv2.imread(image_file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)
    
    # Only proceed if pose landmarks were detected
    if result.pose_landmarks:
       # annotated_image = draw_landmarks_on_image(image_rgb, result)
       # cv2.imshow('Annotated Image', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
       # cv2.waitKey(0)
        # Save landmarks to CSV
        save_landmarks_to_csv(image_file_path, result.pose_landmarks, label)
    else:
        print(f"No landmarks detected for image: {image_file_path}")

# Function to process all images in a directory
def process_directory(directory_path, pose):
    for sub_dir in os.listdir(directory_path):
        sub_dir_path = os.path.join(directory_path, sub_dir)
        if os.path.isdir(sub_dir_path):
            for sub_sub_dir in os.listdir(sub_dir_path):
                sub_sub_dir_path = os.path.join(sub_dir_path, sub_sub_dir)
                if os.path.isdir(sub_sub_dir_path):
                    label = 'non_fall' if 'non_fall' in sub_sub_dir_path else 'fall'
                    for file_name in os.listdir(sub_sub_dir_path):
                        file_path = os.path.join(sub_sub_dir_path, file_name)
                        if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            print(f"Processing file: {file_path}")
                            process_image_file(file_path, pose, label)


# Main function to start the processing
def main():
    root = Tk()
    root.withdraw()
    directory_path = filedialog.askdirectory()
    if directory_path:
        print(f"Selected directory: {directory_path}")
        with create_pose_object() as pose:
            process_directory(directory_path, pose)
    else:
        print("No directory selected")

if __name__ == "__main__":
    main()
