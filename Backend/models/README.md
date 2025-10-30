# Pre-trained Model Weights

## Overview

This directory contains the pre-trained model weights for the deepfake detection ensemble. All model files are tracked using **Git LFS** (Large File Storage).

## Available Models

| Model             | File                    | Size    | Input Size | Parameters |
| ----------------- | ----------------------- | ------- | ---------- | ---------- |
| ResNet50          | `resnet50.pth`          | ~98 MB  | 224×224    | ~25.6M     |
| ResNet152V2       | `resnet152v2.pth`       | ~236 MB | 224×224    | ~60.2M     |
| InceptionResNetV2 | `inceptionresnetv2.pth` | ~215 MB | 299×299    | ~55.8M     |
| Xception          | `xception.pth`          | ~88 MB  | 299×299    | ~22.9M     |
| EfficientNetB4    | `efficientnetb4.pth`    | ~75 MB  | 380×380    | ~19.3M     |
| **Total**         | -                       | ~712 MB | -          | ~183.4M    |

## Model Performance

Training and validation metrics for all models:

| Model                 | Train Accuracy | Train Loss | Val Accuracy | Val Loss |
| --------------------- | -------------- | ---------- | ------------ | -------- |
| **EfficientNetB4**    | **99.82%**     | 0.0050     | 98.68%       | 0.0787   |
| **InceptionResNetV2** | **99.62%**     | 0.0093     | **98.88%**   | 0.0441   |
| **ResNet152V2**       | **99.72%**     | 0.0085     | 98.87%       | 0.0506   |
| **Xception**          | 99.41%         | 0.0048     | 98.79%       | 0.0658   |
| **ResNet50**          | 99.18%         | 0.0206     | 98.57%       | 0.0438   |
| **Average**           | **99.55%**     | 0.0096     | **98.76%**   | 0.0566   |

**Key Highlights**:

- ✅ All models achieve >98.5% validation accuracy
- ✅ Average validation accuracy: **98.76%**
- ✅ Best validation performer: **InceptionResNetV2** (98.88%)
- ✅ Highest training accuracy: **EfficientNetB4** (99.82%)
- ✅ Lowest training loss: **Xception** (0.0048)
- ✅ Minimal overfitting: average generalization gap of 0.79%

## Git LFS Setup

These model files are stored using Git LFS. When you clone this repository:

### Automatic Download (If Git LFS is installed)

If you have Git LFS installed before cloning, the model files will be downloaded automatically:

```bash
# Install Git LFS first
git lfs install

# Then clone the repository
git clone <repository-url>
cd "ML Project/Backend"
```

### Manual Download (If you cloned without Git LFS)

If you cloned the repository without Git LFS, you'll have pointer files instead of actual models. To download them:

```bash
# Install Git LFS
git lfs install

# Pull the actual model files
git lfs pull
```

### Verify Downloads

Check that model files are actual binary files (not text pointers):

```bash
# Linux/Mac
file models/*.pth

# Should show: "models/resnet50.pth: data"
# NOT: "models/resnet50.pth: ASCII text"

# Windows (PowerShell)
Get-ChildItem models/*.pth | ForEach-Object {
    "{0}: {1:N2} MB" -f $_.Name, ($_.Length / 1MB)
}
```

## Installing Git LFS

If you don't have Git LFS installed:

### Windows

```bash
# Using Git for Windows (includes Git LFS)
# Download from: https://git-scm.com/download/win

# Or using Chocolatey
choco install git-lfs

# Or using Scoop
scoop install git-lfs
```

### Linux

```bash
# Debian/Ubuntu
sudo apt-get install git-lfs

# Fedora
sudo dnf install git-lfs

# Arch Linux
sudo pacman -S git-lfs
```

### macOS

```bash
# Using Homebrew
brew install git-lfs

# Using MacPorts
sudo port install git-lfs
```

After installation:

```bash
git lfs install
```

## Model Architecture Details

### ResNet50 & ResNet152V2

- **Architecture**: Deep Residual Networks
- **Modifications**: Final fully-connected layer replaced for binary classification
- **Checkpoint**: Fine-tuned on deepfake dataset

### Xception

- **Architecture**: Depthwise Separable Convolutions
- **Implementation**: Custom implementation in `xception.py`
- **Modifications**: Binary classification head
- **Checkpoint**: Fine-tuned on deepfake dataset

### EfficientNetB4

- **Architecture**: Efficient compound scaling
- **Modifications**: Classifier adapted for binary classification
- **Checkpoint**: Fine-tuned on deepfake dataset

## Training Information

All models were fine-tuned on the 140k Real and Fake Faces dataset:

- **Training Set**: 140,002 images
- **Validation Set**: 39,428 images
- **Test Set**: 10,905 images
- **Optimization**: Adam/AdamW optimizer
- **Loss Function**: Cross-Entropy Loss
- **Data Augmentation**: Random flips, rotations, color jittering

## Usage

The models are automatically loaded by the `ModelManager` class in `service.py`:

```python
from service import ModelManager

# Initialize manager
manager = ModelManager()

# Load all models
manager.load_models()

# Or load specific models
manager.load_models(names=['resnet50', 'xception'])

# Make predictions
result = manager.predict_ensemble(image)  # Ensemble prediction
result = manager.predict_single('resnet50', image)  # Single model
```

## File Integrity

After downloading, verify the files are not corrupted:

```bash
# Each .pth file should be between 75-240 MB
ls -lh models/*.pth

# Python verification
python -c "import torch; torch.load('models/resnet50.pth'); print('OK')"
```

## Troubleshooting

### Issue: Files are text pointers instead of binary

**Solution**:

```bash
git lfs install
git lfs pull
```

### Issue: "This exceeds GitHub's file size limit"

**Solution**: This is normal - the files are stored in Git LFS, not directly in Git. Make sure Git LFS is installed and configured.

### Issue: Out of memory when loading models

**Solution**: Load models individually instead of all at once:

```bash
export LOAD_MODELS=resnet50,xception  # Load only 2 models
uvicorn main:app --reload
```

### Issue: Model files not found

**Solution**: Ensure you're in the correct directory and have pulled LFS files:

```bash
cd Backend
git lfs pull
ls -lh models/*.pth
```

## Alternative: Download from Hugging Face

If you're having issues with Git LFS, you can download models from Hugging Face:

```bash
# Coming soon - models will be hosted on Hugging Face Hub
# huggingface-cli download username/deepfake-models --local-dir models/
```

## License

The model weights are licensed under the same license as the project. See the main LICENSE file for details.

## Contributing

If you train improved models or variants, please consider contributing them back:

1. Ensure the model follows the naming convention
2. Test it with the inference pipeline
3. Submit a pull request with the model file (Git LFS will handle it)
4. Include training details in your PR description

---

**Note**: Git LFS has bandwidth and storage quotas on GitHub. If you encounter quota issues, please open an issue, and we'll provide alternative download links.
