import torch
import torchvision
import torchaudio
import os
from ultralytics import YOLO



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model = YOLO("models/model1/best.pt").to(device)

input_folder_path = "input_videos/train/A1606b0e6_0"

for filename in os.listdir(input_folder_path):
    if filename.endswith(".mp4"):
        input_video_path = os.path.join(input_folder_path, filename)
        
        # Perform the prediction and save the results
        results = model.predict(input_video_path, save=True, device='cuda')

        print(f'Finished for {filename}:')

