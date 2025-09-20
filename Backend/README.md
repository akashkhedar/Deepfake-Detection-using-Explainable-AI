# Deep Fake Detection using ResNet18

A PyTorch-based deep learning project for classifying images as real or AI-generated (fake) using a fine-tuned ResNet18 model. This project includes comprehensive tools for dataset management, model training, testing, and visualization.

## 🎯 Project Overview

This project implements a binary classification system to distinguish between real and AI-generated images. It uses transfer learning with a pre-trained ResNet18 model, fine-tuned on a custom dataset of real and fake images.

### Key Features

- **Binary Classification**: Real vs Fake image detection
- **Transfer Learning**: Leverages pre-trained ResNet18 with ImageNet weights
- **Comprehensive Testing**: Includes confusion matrix, classification reports, and error analysis
- **Grad-CAM Visualization**: Visual explanations of model predictions
- **Dataset Management**: Tools for dataset validation, sampling, and mini-dataset creation
- **Mixed Precision Training**: Optimized training with automatic mixed precision

## 📊 Dataset Structure

The project expects the following dataset structure:

```
dataset/
├── train/
│   ├── Fake/          # AI-generated images
│   └── Real/          # Real images
├── validation/
│   ├── Fake/
│   └── Real/
└── test/
    ├── Fake/
    └── Real/
```

### Dataset Statistics

Based on the included `dataset_summary.json`:

| Split      | Real Images | Fake Images | Total       |
| ---------- | ----------- | ----------- | ----------- |
| Train      | 70,001      | 70,001      | 140,002     |
| Validation | 19,787      | 19,641      | 39,428      |
| Test       | 5,413       | 5,492       | 10,905      |
| **Total**  | **95,201**  | **95,134**  | **190,335** |

- **Image Format**: All images are 256x256 pixels
- **File Types**: JPG format
- **No Corrupt Files**: Dataset integrity verified

## 🚀 Quick Start

### Prerequisites

Ensure you have Python 3.8+ installed with the following packages:

```bash
pip install torch torchvision torchaudio
pip install scikit-learn matplotlib seaborn
pip install Pillow tqdm numpy
pip install torchcam
```

### Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd ML-Project
```

2. Install dependencies:

```bash
pip install -r requirements.txt  # Create this file based on imports
```

3. Verify dataset structure:

```bash
python check_dataset.py --root dataset --make_sample_grid
```

### Training the Model

#### Option 1: Quick Training (Recommended for Testing)

Train on a smaller subset using the mini dataset:

```bash
# Create mini dataset (2000 images per class per split)
python make_mini.py --src dataset --dst dataset_mini --per_class 2000

# Train on mini dataset
python train_resnet18.py
```

#### Option 2: Full Training

For full dataset training, modify `DATA_DIR` in `train_resnet18.py`:

```python
DATA_DIR = "dataset"  # Change from "dataset_mini"
```

Then run:

```bash
python train_resnet18.py
```

### Testing the Model

```bash
python test_resnet18.py
```

This will generate:

- Classification report with precision, recall, F1-score
- Confusion matrix visualization
- Misclassified images analysis
- Grad-CAM visualizations for error analysis

## 🔧 Configuration

### Training Parameters

Key parameters in `train_resnet18.py`:

```python
DATA_DIR = "dataset_mini"    # Dataset path
BATCH_SIZE = 16              # Batch size
NUM_EPOCHS = 8               # Number of training epochs
LR = 1e-4                    # Learning rate
DEVICE = "cuda" if available # Training device
```

### Data Augmentation

Training augmentations:

- Resize to 224x224
- Random horizontal flip
- Color jitter (brightness, contrast, saturation, hue)
- ImageNet normalization

Validation/Test transforms:

- Resize to 224x224
- ImageNet normalization

## 📁 File Structure

```
.
├── train_resnet18.py          # Main training script
├── test_resnet18.py           # Model evaluation and testing
├── check_dataset.py           # Dataset validation and analysis
├── make_mini.py               # Create smaller dataset subset
├── dataset_summary.json       # Dataset statistics and metadata
├── resnet18_best.pth          # Trained model weights (44.8 MB)
├── dataset/                   # Full dataset
├── dataset_mini/              # Subset dataset for quick training
└── README.md                  # This file
```

## 🧠 Model Architecture

- **Base Model**: ResNet18 pre-trained on ImageNet
- **Modification**: Final fully connected layer replaced for binary classification
- **Input Size**: 224x224x3 RGB images
- **Output**: 2 classes (Real=0, Fake=1)
- **Optimizer**: AdamW with weight decay (1e-2)
- **Loss Function**: CrossEntropyLoss
- **Training**: Mixed precision training with GradScaler

## 📈 Model Performance

The trained model (`resnet18_best.pth`) achieves competitive performance on the fake vs real image classification task. Detailed metrics are generated during testing including:

- Per-class precision, recall, and F1-score
- Overall accuracy
- Confusion matrix
- Misclassification analysis

## 🔍 Dataset Analysis Tools

### 1. Dataset Checker (`check_dataset.py`)

Comprehensive dataset validation tool:

```bash
python check_dataset.py --root dataset --sample_per_class 12 --make_sample_grid
```

Features:

- Counts images per class and split
- Detects corrupt/invalid images
- Analyzes image dimensions
- Creates sample grids for visual inspection
- Generates detailed JSON summary

### 2. Mini Dataset Creator (`make_mini.py`)

Creates smaller dataset subsets for faster experimentation:

```bash
python make_mini.py --src dataset --dst dataset_mini --per_class 2000
```

## 🎨 Visualization Features

### Grad-CAM Analysis

The testing script includes Grad-CAM (Gradient-weighted Class Activation Mapping) for:

- Visualizing which image regions the model focuses on
- Understanding model decision-making process
- Analyzing misclassified examples
- Improving model interpretability

### Sample Grids

Generated sample grids (`sample_grid_*.jpg`) provide quick visual overview of:

- Dataset diversity
- Image quality
- Class distribution
- Potential data issues

## 🛠️ Development Notes

### Windows Compatibility

- Uses `num_workers=0` in DataLoader for Windows stability
- PowerShell-compatible commands
- Windows path handling in dataset scripts

### Memory Optimization

- Mixed precision training reduces memory usage
- Configurable batch sizes
- Efficient data loading with proper transforms

### Extensibility

The codebase is designed for easy extension:

- Modular design with clear separation of concerns
- Configurable parameters
- Well-documented functions
- Standard PyTorch patterns

## 📋 Usage Examples

### Basic Training

```bash
# Quick start with mini dataset
python make_mini.py
python train_resnet18.py
python test_resnet18.py
```

### Dataset Analysis

```bash
# Full dataset analysis
python check_dataset.py --root dataset --make_sample_grid --out analysis.json

# Check mini dataset
python check_dataset.py --root dataset_mini --make_sample_grid --out mini_analysis.json
```

### Custom Training

```python
# Modify training parameters in train_resnet18.py
DATA_DIR = "your_dataset_path"
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 5e-5
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Dependencies

- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **scikit-learn**: Machine learning metrics
- **matplotlib/seaborn**: Visualization
- **Pillow**: Image processing
- **tqdm**: Progress bars
- **numpy**: Numerical computing
- **torchcam**: Grad-CAM implementation

## 📞 Support

For questions, issues, or contributions, please:

1. Check existing issues
2. Create a new issue with detailed description
3. Include error messages and system information
4. Provide minimal reproducible examples

---

_This project demonstrates modern deep learning practices for image classification with comprehensive tooling for dataset management, model training, and result analysis._
