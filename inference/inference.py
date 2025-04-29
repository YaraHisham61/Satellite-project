import numpy as np
import pandas as pd
import cv2
import torch
from model_arch.cloud_net import model_arch
import pandas.api.types
from pathlib import Path


MODEL_PATH = "best_cloud_net_0.8816.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass

def dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes the Dice coefficient between two binary masks."""
    intersection = np.sum(mask1 * mask2)
    return (2.0 * intersection) / (np.sum(mask1) + np.sum(mask2) + 1e-7)  # Avoid division by zero

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """Computes the Dice score between solution and submission."""
    
    # Check if required columns exist
    required_columns = {row_id_column_name, "segmentation"}
    if not required_columns.issubset(solution.columns) or not required_columns.issubset(submission.columns):
        raise ParticipantVisibleError("Solution and submission must contain 'id' and 'segmentation' columns")
    
    # Ensure the IDs match between solution and submission
    if not solution[row_id_column_name].equals(submission[row_id_column_name]):
        raise ParticipantVisibleError("Submission IDs do not match solution IDs")
    
    # Delete the row ID column as Kaggle aligns solution and submission before passing to score()
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    # Decode RLE masks and compute Dice score
    dice_scores = []
    for solution_seg, submission_seg in zip(solution["segmentation"], submission["segmentation"]):
        solution_mask = rle_decode(solution_seg)
        submission_mask = rle_decode(submission_seg)
        dice_scores.append(dice_coefficient(solution_mask, submission_mask))
    
    return np.mean(dice_scores)

def rle_decode(mask_rle, shape = (512, 512)):
    """
    Decodes an RLE-encoded string into a binary mask.
    
    Args:
        mask_rle (str): RLE-encoded string.
        shape (tuple): (height, width) of the output mask.
    
    Returns:
        np.ndarray: Decoded binary mask.
    """
    if not mask_rle:
        return np.zeros(shape, dtype=np.uint8)

    s = list(map(int, mask_rle.split()))
    starts, lengths = s[0::2], s[1::2]  # Separate start positions and lengths

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)  # Create a flat mask
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1  # Fill mask with 1s

    return mask.reshape(shape, order='F')  # Reshape in column-major order

def resize_rle(rle_string, original_shape=(512, 512), target_shape=(256, 256)):
    """
    Decode an RLE string, resize the mask, and re-encode to RLE.
    
    Args:
        rle_string (str): RLE-encoded string for original mask.
        original_shape (tuple): Shape of original mask (height, width).
        target_shape (tuple): Desired shape of resized mask (height, width).
    
    Returns:
        str: RLE-encoded string for resized mask.
    """
    # Decode RLE to 512x512 mask
    mask = rle_decode(rle_string, shape=original_shape)
    
    # Resize to 256x256 using nearest-neighbor interpolation
    resized_mask = cv2.resize(mask, target_shape[::-1], interpolation=cv2.INTER_NEAREST)
    
    # Ensure binary (0, 1)
    resized_mask = (resized_mask > 0).astype(np.uint8)
    
    # Re-encode to RLE
    return rle_encode(resized_mask) 

def rle_decode(mask_rle: str, shape=(256, 256)) -> np.ndarray:
    """Decodes an RLE-encoded string into a binary mask with validation checks."""
    
    if not isinstance(mask_rle, str) or not mask_rle.strip() or mask_rle.lower() == 'nan':
        # Return all-zero mask if RLE is empty, invalid, or NaN
        return np.zeros(shape, dtype=np.uint8)
    
    try:
        s = list(map(int, mask_rle.split()))
    except:
        raise Exception("RLE segmentation must be a string and containing only integers")
    
    if len(s) % 2 != 0:
        raise Exception("RLE segmentation must have even-length (start, length) pairs")
    
    if any(x < 0 for x in s):
        raise Exception("RLE segmentation must not contain negative values")
    
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    starts, lengths = s[0::2], s[1::2]
    
    for start, length in zip(starts, lengths):
        if start >= mask.size or start + length > mask.size:
            raise Exception("RLE indices exceed image size")
        mask[start:start + length] = 1
    
    return mask.reshape(shape, order='F')  # Convert to column-major order

def process_csv_rle(input_csv, output_csv, original_shape=(512, 512), target_shape=(256, 256)):
    """
    Read a CSV with RLE-encoded masks, resize masks, and save to a new CSV.
    
    Args:
        input_csv (str or Path): Path to input CSV with 'id' and 'segmentation' columns.
        output_csv (str or Path): Path to output CSV.
        original_shape (tuple): Shape of original masks (height, width).
        target_shape (tuple): Shape of resized masks (height, width).
    """
    # Read CSV, ensuring segmentation is string and NaN is ''
    df = pd.read_csv(input_csv, dtype={'id': str, 'segmentation': str}).fillna({'segmentation': ''})

    # Resize each RLE
    df['segmentation'] = df['segmentation'].apply(
        lambda rle: resize_rle(rle, original_shape, target_shape)
    )
    
    # Save to new CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Resized masks saved to: {output_csv}")

def main():

    model_loaded = model_arch(num_of_channels=4, num_of_classes=1).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device,weights_only = True)
    model_loaded.load_state_dict(state_dict)
    model_loaded.eval()

    # Paths to input and output CSVs
    pred_csv = "output.csv"  
    original_csv = "original_masks.csv"  
    pred_resized_csv = "output_resized.csv"
    original_resized_csv = "original_masks_resized.csv"

    # Process CSVs to resize masks
    process_csv_rle(pred_csv, pred_resized_csv)
    process_csv_rle(original_csv, original_resized_csv)


    df_pred = pd.read_csv(pred_resized_csv,dtype={'id': int, 'segmentation': str})
    df_original = pd.read_csv(original_resized_csv,dtype={'id': int, 'segmentation': str})

    print(score(df_original, df_pred, "id"))



if __name__ == "__main__":
    main()

    