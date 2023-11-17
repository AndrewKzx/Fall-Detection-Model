from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('my_model.keras')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the scaler parameters
scaler_params = np.load('scaler_params.npz')
scaler = MinMaxScaler()
scaler.min_ = scaler_params['min_']
scaler.scale_ = scaler_params['scale_']

# Function to calculate features for a single frame
def calculate_frame_features(pose_landmarks):
    # Extract coordinates of relevant landmarks
    head = np.array([pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,
                     pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,
                     pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].z])
    
    hip_center = np.array([(pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x + 
                            pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2,
                           (pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y + 
                            pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2,
                           (pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z + 
                            pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z) / 2])
    
    shoulder_center = np.array([(pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x + 
                                 pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2,
                                (pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y + 
                                 pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2,
                                (pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z + 
                                 pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z) / 2])
    
    knee_center = np.array([(pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x + 
                             pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x) / 2,
                            (pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y + 
                             pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y) / 2,
                            (pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z + 
                             pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].z) / 2])

    # Compute additional features
    distance_head_to_ground = head[1]  # distance_head_to_ground is the y coordinate of the head
    v_torso = shoulder_center - hip_center
    v_leg = knee_center - hip_center
    cos_theta = np.dot(v_torso, v_leg) / (np.linalg.norm(v_torso) * np.linalg.norm(v_leg))
    angle_torso_leg = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * (180 / np.pi)  # Convert radians to degrees
    
    # Return a numpy array of features
    return np.array([[distance_head_to_ground, angle_torso_leg]])



# Video Stream Generator Function
def generate_frames():
    cap = cv2.VideoCapture(0)  # Use 0 for web camera
    sequence = []  # Store a sequence of frames
    fall_detected_frames = 0  # Counter for consecutive fall detected frames
    FALL_DETECTED_CONSECUTIVE_FRAMES = 10  # Number of consecutive frames to confirm a fall
    FALL_CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold to determine a fall
    prediction = None  
    
    saved_frame_counter = 0

    # Process each frame
    while True:
        success, frame = cap.read()  # Read the camera frame
        if not success:
            break
        else:
            # Convert the image to RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_image)

            # If pose landmarks were detected, calculate features
            if result.pose_landmarks:
                # Draw the pose landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

                # Save the first frame to a file for inspection
                if saved_frame_counter < 1:
                    cv2.imwrite('debug_frame.jpg', frame)
                    saved_frame_counter += 1
                
                frame_features = calculate_frame_features(result.pose_landmarks)
                sequence.append(frame_features)
                
                # Check if we have enough frames for a sequence
                if len(sequence) == 10:
                    print("Predicting...")
                    sequence_array = np.array(sequence).reshape(1, 10, 2)
                    prediction = model.predict(sequence_array)
                    sequence.pop(0)  # Remove the oldest frame from the sequence
                    print("Prediction:", prediction)
                    

            # Annotate the frame with the prediction result
            if prediction is not None and prediction[0][1] > FALL_CONFIDENCE_THRESHOLD:
                fall_detected_frames += 1
                if fall_detected_frames >= FALL_DETECTED_CONSECUTIVE_FRAMES:
                    cv2.putText(frame, 'Fall Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                fall_detected_frames = 0
                cv2.putText(frame, 'No Fall Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag:
                continue
            
            # Yield the output frame in byte format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + b'\r\n')

# Flask routes and the rest of the Flask app setup would go here...

# Define the root route to render the index.html template
@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

# Define the route for video feed
@app.route('/video_feed')
def video_feed():
    # Return the video stream response
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
