import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from model_arch.cloud_net_deep_supervision import model_arch
from datasets.inference_dataset import InferenceDataset
from skimage.transform import resize
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler, schedule

MODEL_PATH = "models/checkpoint_epoch_45.pth"
IMG_DIR = "testset"
SUBMISSION_FILE = "testset/sample_submission.csv"
OUT_DIR = "output"
BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
img_names = [x.name for x in sorted(Path(IMG_DIR).glob("*.tif"))]
print(f"Found {len(img_names)} images in {IMG_DIR}")

def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).    
    Args:
        mask (np.ndarray): 2D binary mask (0s and 1s).
    Returns:
        str: RLE-encoded string, or a single space " " if mask is all zeros.
    """
    if np.sum(mask) == 0:
        return " "  # As it seems that kaggle reject nulls. We'll handle cloud-free images with empty spaces.
    
    pixels = mask.flatten(order='F')  # Flatten in column-major order
    pixels = np.concatenate([[0], pixels, [0]])  # Add padding to detect transitions
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # Get transition indices
    runs[1::2] -= runs[::2]  # Compute run lengths
    runs[::2] -= 1  # Make it 0-indexed instead of 1-indexed

    return " ".join(map(str, runs))  # Convert to string format


def save_predictions(model, data_loader, device, output_dir, thresh=0.5, save_images=True):
    """
    Save model predictions as both RLE CSV and PNG masks
    """
    model.eval()
    output_dir = Path(output_dir)
    
    df_submission = pd.read_csv(SUBMISSION_FILE,dtype={'id': str, 'segmentation': str})
    pred_dict = {str(row['id']): '' for _, row in df_submission.iterrows()}
    profiler_schedule = schedule(wait=1, warmup=1, active=3, repeat=1)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        schedule=profiler_schedule,
        on_trace_ready=tensorboard_trace_handler(output_dir / "profiler"),
    ) as prof:
        with torch.no_grad():
            for images, img_names in tqdm(data_loader, desc="Processing images"):
                images = images.to(device)
                with prof:
                    
                    outputs = model(images)['final'].sigmoid()
                    pred_masks = (outputs > thresh).float().cpu().numpy()
                
                for i, img_name in enumerate(img_names):
                    img_id = Path(img_name).stem
                    mask = pred_masks[i, 0]  # To remove channel dim
                    if save_images:
                        (output_dir / "masks").mkdir(parents=True, exist_ok=True)
                        mask_uint8 = (mask * 255).astype(np.uint8)
                        cv2.imwrite(str(output_dir / "masks" / f"{img_id}.png"), mask_uint8)
                    mask = resize(mask, (256, 256), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                    rle = rle_encode(mask) if mask.any() else ' '
                    pred_dict[img_id] = rle
            prof.step()

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # Save CSV
    df_submission['segmentation'] = df_submission['id'].map(pred_dict)
    df_submission['segmentation'] = df_submission['segmentation'].fillna(" ")
    df_submission.to_csv(Path(output_dir)/'predictions.csv', index=False)
    print(f"Saved {len(df_submission)} predictions to {output_dir}") 


def main():

    model_loaded = model_arch(num_of_channels=4, num_of_classes=1).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device,weights_only = True)
    model_loaded.load_state_dict(state_dict)
    model_loaded.eval()
    print(f"loaded from {MODEL_PATH}")
    test_dataset = InferenceDataset(IMG_DIR, img_names)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False )
    save_predictions(
        model=model_loaded,
        data_loader=test_loader,
        device=device,
        output_dir=OUT_DIR,
        thresh=0.64,
        save_images=False
    )


if __name__ == "__main__":
        main()