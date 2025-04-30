from pathlib import Path
import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_names, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.img_names = img_names
        self.transform = transform
        
        self.means = np.array([2837.04085609, 2831.98116589, 2734.94209197, 3677.72763665], dtype=np.float16)
        self.stds = np.array([3190.10911234, 2927.71436665, 2814.50304193 ,2432.04465005], dtype=np.float16)  # Replace with real stds!
    
    def __len__(self):
        return len(self.img_names)
        
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = self.img_dir / img_name
        mask_path = self.mask_dir / img_name
        
        try:
            with rasterio.open(img_path) as src:
                image = src.read()  # Shape: [4, H, W]
                image = image.astype(np.float32)  # Use float32 for stability
                
                # Normalize
                image = (image - self.means[:, None, None]) / self.stds[:, None, None]
                
                # Convert to tensor and ensure channel-first
                image = torch.from_numpy(image)  # [4, H, W]
    
            with rasterio.open(mask_path) as src:
                mask = (src.read(1) > 0).astype(np.float32)
                mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
    
            return image, mask
    
        except Exception as e:
            print(f"Error loading {img_name}: {str(e)}")
            return torch.zeros((4, 512, 512)), torch.zeros((1, 512, 512))

    def get_name(self, idx):
        """Returns the filename for a given index"""
        return self.img_names[idx]