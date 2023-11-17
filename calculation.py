# calculation.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('C:\\Users\\khorz\\Desktop\\Capstone 2\\landmarks.csv')

# Display the first few rows of the data
print(data.head())

# Group the data by frame and label
grouped = data.groupby(['file_name', 'label'], as_index=False)

def check_landmark(group, landmark_id):
    landmark_data = group[group['landmark_id'] == landmark_id]
    if landmark_data.empty:
        return None
    return landmark_data.iloc[0][['x', 'y', 'z']].values

def calculate_features(group):
    # Initialize the DataFrame to store features
    features_df = group.copy()
    
    # Extract coordinates of relevant landmarks
    head = check_landmark(group, 0)  # Assuming landmark 0 is the head
    hip_center = check_landmark(group, 11)
    shoulder_center = check_landmark(group, 12)
    knee_center = check_landmark(group, 13)

    # If any landmark is missing, return the group without additional features
    if any(x is None for x in [head, hip_center, shoulder_center, knee_center]):
        return features_df
    
    # Compute additional features
    distance_head_to_ground = head[1]
    v_torso = shoulder_center - hip_center
    v_leg = knee_center - hip_center
    cos_theta = np.dot(v_torso, v_leg) / (np.linalg.norm(v_torso) * np.linalg.norm(v_leg))
    angle_torso_leg = np.arccos(cos_theta) * (180 / np.pi)  # Convert radians to degrees
    
    # Append additional features to the features DataFrame
    features_df['distance_head_to_ground'] = distance_head_to_ground
    features_df['angle_torso_leg'] = angle_torso_leg
    
    return features_df

# Calculate features for each frame
features = grouped.apply(calculate_features)

# Flatten the hierarchical index resulting from groupby
features.reset_index(drop=True, inplace=True)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Select columns for scaling
columns_to_scale = features.columns.difference(['file_name', 'label', 'landmark_id'])

# Fit the scaler on the features and transform
features[columns_to_scale] = scaler.fit_transform(features[columns_to_scale])

# Save the scaler parameters
np.savez('scaler_params.npz', min_=scaler.min_, scale_=scaler.scale_)

# Save the combined features to a CSV file
features.to_csv('output.csv', index=False)
