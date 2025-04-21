import os
from zipfile import ZipFile

# Set Kaggle API credentials
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

# Make sure your kaggle.json is already in the current dir
os.system("mv kaggle.json ~/.kaggle/")
os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)

# Download the dataset
os.system("kaggle datasets download -d ritikagiridhar/2000-hand-gestures")

# Create project data directory
project_data_path = "project_dir/hand_gestures"
os.makedirs(project_data_path, exist_ok=True)

# Unzip the downloaded file
with ZipFile("2000-hand-gestures.zip", 'r') as zip_ref:
    zip_ref.extractall(project_data_path)

print(f"✅ Dataset extracted to {project_data_path}")