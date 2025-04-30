import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import rasterio

class InferenceDataset(Dataset):
    """Dataset for inference with proper normalization and error handling"""
    def __init__(self, img_dir, img_names,size = (512,512)):
        self.img_dir = Path(img_dir)
        self.img_names = img_names
        self.size = size
        self.means = np.array([2837.04085609, 2831.98116589, 2734.94209197, 3677.72763665], dtype=np.float32)
        self.stds = np.array([3190.10911234, 2927.71436665, 2814.50304193, 2432.04465005], dtype=np.float32)
    
    def __len__(self):
        return len(self.img_names)
        
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = self.img_dir / img_name
        
        try:
            with rasterio.open(img_path) as src:
                image = src.read()  # Shape: [4, H, W]
                image = image.astype(np.float32)
                image = (image - self.means[:, None, None]) / self.stds[:, None, None]
            # Resize all channels
                image = np.stack([
                    cv2.resize(ch, self.size[::-1]) for ch in image
                ])
                return torch.from_numpy(image), img_name  # Return both image and name
            
        except Exception as e:
            print(f"Error loading {img_name}: {str(e)}")
            return torch.zeros((4, 512, 512)), img_name  # Return zero tensor with name