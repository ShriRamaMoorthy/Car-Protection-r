import os
import datetime

def parse_timestamp(filename):
    timestamp_str = filename.split('_')[-2] + '_' + filename.split('_')[-1].split('.')[0]
    return datetime.datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")

current_timestamp = datetime.datetime.now()
files = os.listdir()
model_files = [file for file in files if file.startswith('faces_embeddings_done_4classes_') and file.endswith('.npz')]
print(model_files)
closest_file = None
closest_difference = float('inf')

for file in model_files:
    timestamp = parse_timestamp(file)
    time_difference = abs(current_timestamp - timestamp).total_seconds()
    if time_difference < closest_difference:
        closest_file = file
        closest_difference = time_difference

# Load the closest model
if closest_file:
    print(f"Loaded model from: {closest_file}")
else:
    print("No model files found.")
