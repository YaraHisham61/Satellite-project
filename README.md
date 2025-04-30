# Satellite Imaging Project: AI - Cloud Masking

## Installation

### Prerequisites
- Python 3.8+
- numpy==2.2.5
- opencv_python==4.11.0.86
- pandas==2.2.3
- rasterio==1.4.3
- skimage==0.0
- torch==2.6.0
- tqdm==4.67.1
- NVIDIA GPU (optional, for acceleration)

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/YaraHisham61/Satellite-project.git
   cd Satellite-project
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt

## Project Structure
```
├── datasets/                  # Dataset loaders and utilities
│   ├── custom_dataset.py      # Custom dataset class
│   └── inference_dataset.py   # Inference-specific data handler
│
├── inference/                 # Model evaluation scripts
│   ├── run_inference.py       # Main inference pipeline
│   └── rle-encoder-decoder.py # RLE encoding/decoding tools
│
├── models/                    # Pre-trained model checkpoints
│   ├── checkpoint_epoch_25.pth
│   └── checkpoint_epoch_45.pth
│
├── model_arch/                # Model architecture definitions
│   ├── cloud_net.py           # Base CloudNet model
│   ├── cloud_net_attention.py # Attention variant
│   └── cloud_net_deep_supervision.py
│
├── notebooks/                 # Jupyter notebooks (Kaggle training, EDA)
├── output/                    # Inference outputs
│   ├── inference_trace.json   # Runtime logs
│   └── predictions.csv        # Prediction results
│
├── report/                    # Project documentation
│   ├── report.docx            # Editable report
│   └── report.pdf            # Final deliverable
│
└── requirements.txt           # Python dependencies
