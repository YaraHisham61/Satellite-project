import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model_arch.cloud_net_deep_supervision import model_arch
from pathlib import Path
import rasterio


MODEL_PATH = "best_cloud_net_0.8816.pth"
IMG_DIR = "test_images"
OUT_DIR = "output"
SUBMISSION_FILE = "sample_submission.csv"
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InferenceDataset(Dataset):
    def __init__(self, img_dir,img_names):
        self.img_dir = Path(img_dir)
        self.img_names = img_names
        self.means = np.array([2837.04085609, 2831.98116589, 2734.94209197, 3677.72763665], dtype=np.float16)
        self.stds = np.array([3190.10911234, 2927.71436665, 2814.50304193 ,2432.04465005], dtype=np.float16)
    
    def __len__(self):
        return len(self.img_names)
        
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = self.img_dir / img_name
        
        
        try:
            with rasterio.open(img_path) as src:
                image = src.read()  # Shape: [4, H, W]
                image = image.astype(np.float32)  # Use float32 for stability
                
                # Normalize
                image = (image - self.means[:, None, None]) / self.stds[:, None, None]
                
                # Convert to tensor and ensure channel-first
                image = torch.from_numpy(image)  # [4, H, W]

            return image
    
        except Exception as e:
            print(f"Error loading {img_name}: {str(e)}")
            return torch.zeros((4, 512, 512))

    def get_name(self, idx):
        """Returns the filename for a given index"""
        return self.img_names[idx]
    

def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).
    
    Args:
        mask (np.ndarray): 2D binary mask (0s and 1s).
    
    Returns:
        str: RLE-encoded string.
    """
    pixels = mask.flatten(order='F')  # Flatten in column-major order
    pixels = np.concatenate([[0], pixels, [0]])  # Add padding to detect transitions
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # Get transition indices
    runs[1::2] -= runs[::2]  # Compute run lengths
    runs[::2] -= 1  # Make it 0-indexed instead of 1-indexed
    return " ".join(map(str, runs))  # Convert to string format

def save_outputs(model, data_loader, device, output_dir , thresh=0.5, submission_file=None):
    """Save model predictions with proper handling of dictionary outputs"""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_names = data_loader.dataset.img_names
    pred_results = []
    batch_start_idx = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                pred_masks = outputs['final'].sigmoid() 
            
            pred_masks = (pred_masks > thresh).float()

            for i in range(images.size(0)):
                img_id = Path(img_names[batch_start_idx + i]).stem

                pred_mask = pred_masks[i].squeeze(0).cpu().numpy()
                pred_mask = (pred_mask > 0).astype(np.uint8)
                pred_mask = cv2.resize(pred_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                pred_mask = (pred_mask > 0).astype(np.uint8)
                pred_rle = rle_encode(pred_mask) if pred_mask.any() else ''
                pred_results.append({'id': img_id, 'segmentation': pred_rle})

            batch_start_idx += images.size(0)

    pred_df = pd.DataFrame(pred_results, columns=['id', 'segmentation'])

    if submission_file:
        submission_df = pd.read_csv(submission_file, dtype={'id': str})
        pred_df = submission_df[['id']].merge(pred_df, on='id', how='left')

    pred_csv = output_dir / 'output.csv'
    pred_df.to_csv(pred_csv, index=False)
           

def main():

    model_loaded = model_arch(num_of_channels=4, num_of_classes=1).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device,weights_only = True)
    model_loaded.load_state_dict(state_dict)
    model_loaded.eval()
    img_names =[x.name for x in sorted(Path(IMG_DIR).glob("*.tif"))]
    test_dataset = InferenceDataset(IMG_DIR, img_names=img_names)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    save_outputs(model_loaded, test_loader, device, output_dir= OUT_DIR, thresh = 0.5, submission_file= SUBMISSION_FILE)

if __name__ == "__main__":
    main()

    