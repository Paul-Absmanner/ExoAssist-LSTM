import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Path to the folder containing the files
folder_path = 'C:\\Users\\paula\\Documents\\MEGAsync\\JKU\\5Semester\\Practical_work\\code\\data'

# Get a list of all CSV files in the folder
file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Function to preprocess a single file
def preprocess_file(file_path, sequence_length=50, overlap=25):
    # Load the file
    data = pd.read_csv(file_path)
    
    # Remove "Unnamed" columns
    data_cleaned = data.drop(columns=[col for col in data.columns if "Unnamed" in col])
    
    # Normalize coordinate columns
    coordinate_columns = [col for col in data_cleaned.columns if col not in ['Label', 'Time']]
    scaler = MinMaxScaler()
    data_cleaned[coordinate_columns] = scaler.fit_transform(data_cleaned[coordinate_columns])
    
    # Create time-series sequences
    sequences = []
    labels = []
    times = []  # Initialize times list to track sequence metadata
    file_name = os.path.basename(file_path)
    
    for start in range(0, len(data_cleaned) - sequence_length, sequence_length - overlap):
        end = start + sequence_length
        sequence = data_cleaned.iloc[start:end][coordinate_columns].values
        label = data_cleaned.iloc[start:end]['Label'].mode()[0]
        time = data_cleaned.iloc[start:end]['Time'].values[0]  # Start time of the sequence
        
        sequences.append(sequence)
        labels.append(label)
        times.append({'file': file_name, 'time': time})
    
    return np.array(sequences), np.array(labels), times

# Initialize empty lists to store sequences, labels, and times from all files
all_sequences = []
all_labels = []
all_times = []

# Process each file
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    print(f"Processing file: {file_name}")
    
    sequences, labels, times = preprocess_file(file_path)
    all_sequences.append(sequences)
    all_labels.append(labels)
    all_times.extend(times)

# Combine all sequences, labels, and times from all files
all_sequences = np.concatenate(all_sequences, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Save the combined data
np.save(os.path.join(folder_path, 'features.npy'), all_sequences)
np.save(os.path.join(folder_path, 'labels.npy'), all_labels)
np.save(os.path.join(folder_path, 'file_times.npy'), np.array(all_times, dtype=object))

# Output shapes for verification
print(f"Total Features shape: {all_sequences.shape}")
print(f"Total Labels shape: {all_labels.shape}")
print(f"Total Times length: {len(all_times)}")
