# Deepfake Detection using Explainable AI

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![React](https://img.shields.io/badge/react-19.1.1-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)

An advanced deepfake detection system powered by an ensemble of state-of-the-art deep learning models with explainable AI capabilities. This project combines multiple CNN architectures with GradCAM visualization and AI-powered explanations to detect manipulated images with high accuracy and transparency.

## 🌟 Features

- **Ensemble Model Architecture**: Combines 4 powerful CNN models for robust predictions

  - ResNet50
  - ResNet152V2
  - Xception
  - EfficientNetB4

- **Explainable AI**:

  - GradCAM heatmap visualization showing which image regions influenced the decision
  - AI-generated natural language explanations powered by Google Gemini
  - Per-model and ensemble confidence scores

- **Modern Web Interface**:

  - Responsive React-based UI with Material-UI components
  - Real-time image analysis
  - Interactive heatmap overlays
  - Support for both ensemble and individual model predictions

- **Production-Ready Backend**:
  - FastAPI with async support
  - CORS-enabled for cross-origin requests
  - GPU/CPU support with automatic device detection
  - Robust error handling and logging

## 📊 Dataset

The model is trained on a comprehensive dataset of real and AI-generated images:

- **Training Set**: 140,002 images (70,001 fake + 70,001 real)
- **Validation Set**: 39,428 images (19,641 fake + 19,787 real)
- **Test Set**: 10,905 images (5,492 fake + 5,413 real)
- **Image Resolution**: 256x256 pixels
- **Total**: 190,335 images

All images are verified for integrity with zero corruption.

📥 **Dataset is not included in this repository**. See [`Backend/DATASET.md`](Backend/DATASET.md) for download instructions from Kaggle.

## 🏗️ Project Structure

```
ML Project/
├── .gitattributes                # Git LFS configuration for model files
├── .gitignore                    # Git ignore patterns
├── README.md                     # This file
│
├── Backend/                      # FastAPI backend service
│   ├── main.py                   # FastAPI application & endpoints
│   ├── service.py                # Model management, inference & GradCAM
│   ├── check_dataset.py          # Dataset validation utility
│   ├── requirements.txt          # Python dependencies
│   ├── DATASET.md                # 📥 Dataset download instructions
│   ├── .gitignore                # Backend-specific ignore patterns
│   ├── models/                   # Model weights (Git LFS tracked)
│   │   ├── README.md             # 📥 Model download & LFS guide
│   │   ├── __init__.py
│   │   ├── xception.py           # Custom Xception implementation
│   │   ├── resnet50.pth          # 📦 Git LFS (~98 MB)
│   │   ├── resnet152v2.pth       # 📦 Git LFS (~236 MB)
│   │   ├── xception.pth          # 📦 Git LFS (~88 MB)
│   │   └── efficientnetb4.pth    # 📦 Git LFS (~75 MB)
│   └── dataset/                  # Training data (not in repo)
│       ├── Train/                # Download from Kaggle
│       │   ├── Fake/             # (see DATASET.md)
│       │   └── Real/
│       ├── Validation/
│       │   ├── Fake/
│       │   └── Real/
│       └── Test/
│           ├── Fake/
│           └── Real/
│
└── Frontend/                     # React + Vite frontend
    ├── src/
    │   ├── App.jsx               # Main application component
    │   ├── main.jsx              # React entry point
    │   ├── index.css             # Global styles
    │   └── components/
    │       ├── HomePage.jsx      # Main page with dark theme
    │       ├── Navbar/
    │       │   └── Navbar.jsx    # Navigation bar component
    │       └── ModelSection/
    │           ├── ModelSection.jsx        # Main analysis interface
    │           ├── ImageUploadArea.jsx     # Drag & drop upload
    │           ├── ImageDisplay.jsx        # Original image display
    │           ├── HeatmapDisplay.jsx      # GradCAM visualization
    │           ├── ExplanationBox.jsx      # AI explanation display
    │           ├── RealImageMessage.jsx    # Real image feedback
    │           ├── WaitingState.jsx        # Loading state
    │           ├── index.js                # Component exports
    │           └── hooks/
    │               └── useImageAnalysis.js # Image analysis logic hook
    ├── package.json
    ├── vite.config.js
    └── eslint.config.js
```

## 🚀 Installation

> **📖 For detailed step-by-step instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ and npm/yarn
- **Git LFS** (for downloading model weights) - [Installation Guide](https://git-lfs.com/)
- CUDA-capable GPU (optional, but recommended for faster inference)
- Google Gemini API key (for AI explanations)

### Backend Setup

1. **Clone the repository with Git LFS**:

   ```bash
   # Install Git LFS first (if not already installed)
   git lfs install

   # Clone the repository (this will download model files via LFS)
   git clone <repository-url>
   cd "ML Project"
   ```

   > **Note**: Model files (`.pth`) are stored using Git LFS. If you cloned without Git LFS, run `git lfs pull` to download them. See [`Backend/models/README.md`](Backend/models/README.md) for details.

2. **Download the dataset**:

   The dataset is **not included** in the repository. Follow the instructions in [`Backend/DATASET.md`](Backend/DATASET.md) to download it from Kaggle.

   Quick start:

   ```bash
   # Install Kaggle CLI
   pip install kaggle

   # Configure Kaggle credentials (see DATASET.md)
   # Then download:
   cd Backend
   kaggle datasets download -d xhlulu/140k-real-and-fake-faces
   unzip 140k-real-and-fake-faces.zip -d dataset/
   ```

3. **Navigate to Backend directory**:

   ```bash
   cd Backend
   ```

4. **Create a virtual environment**:

   ```bash
   python -m venv venv

   # Windows
   .\venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

5. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

6. **Configure environment variables**:
   Create a `.env` file in the `Backend` directory:

   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   DEVICE=cuda  # or 'cpu' if no GPU available
   LOAD_MODELS=all  # or comma-separated model names
   ```

### Frontend Setup

1. **Navigate to Frontend directory**:

   ```bash
   cd Frontend
   ```

2. **Install dependencies**:

   ```bash
   npm install
   # or
   yarn install
   ```

3. **Configure API endpoint** (optional):
   Create a `.env` file in the `Frontend` directory:
   ```env
   VITE_API_BASE=http://localhost:8000
   ```

## 🎮 Usage

### Starting the Backend

From the `Backend` directory:

```bash
# Development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

#### API Endpoints

- `GET /` - Health check
- `GET /models/` - List available models
- `GET /status/` - Get model loading status and device info
- `POST /predict/` - Analyze an image
  - Query params: `model` (optional) - specific model name or omit for ensemble
  - Body: multipart/form-data with image file

### Starting the Frontend

From the `Frontend` directory:

```bash
# Development mode
npm run dev
# or
yarn dev

# Production build
npm run build
npm run preview
```

The application will be available at `http://localhost:5173`

### Using the Application

1. **Open the web interface** at `http://localhost:5173`
2. **Select a model**:
   - Choose "Ensemble (All Models)" for combined predictions
   - Or select an individual model (ResNet50, ResNet152, etc.)
3. **Upload an image**:
   - Click "Choose Image" or drag and drop
   - Supported formats: JPG, JPEG, PNG, BMP, WebP, TIFF
4. **View results**:
   - Prediction label (Real/Fake) with confidence score
   - GradCAM heatmap overlay showing attention regions
   - AI-generated explanation of the decision
   - Per-model confidence breakdown (ensemble mode)

## 🔬 Technical Details

### Model Architecture

Each model in the ensemble is a pretrained CNN fine-tuned for binary classification:

- **Input**: RGB images resized to model-specific dimensions

  - ResNet50/152: 224×224
  - Xception: 299×299
  - EfficientNetB4: 380×380

- **Output**: 2-class softmax (Real vs Fake)

- **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Ensemble Strategy

- **Averaging**: Simple average of softmax probabilities across all models
- **Prediction**: argmax of ensemble probabilities
- **Heatmap Fusion**: Weighted average of GradCAM activations based on per-model confidence

### GradCAM Visualization

- Target layers:

  - ResNet models: `layer4[-1].conv2`
  - Xception: `block4`
  - EfficientNetB4: `features[-1]`

- Overlay composition: 40% heatmap + 60% original image

### AI Explanations

- **Model**: Google Gemini 2.5 Flash
- **Input**: Prediction label + GradCAM heatmap overlay (base64 PNG)
- **Output**: 2-3 sentence natural language explanation
- **Fallback**: Generic explanation if API fails

## 📈 Performance

The ensemble approach provides:

- **High Accuracy**: Combining multiple architectures reduces individual model biases
- **Robust Predictions**: Different models capture different fake patterns
- **Confidence Calibration**: Averaging softmax outputs provides well-calibrated probabilities

## 🛠️ Utilities

### Dataset Validation Tool

Validate and analyze the dataset structure:

```bash
cd Backend
python check_dataset.py --root dataset --sample_per_class 12 --out dataset_summary.json --make_sample_grid
```

**Options**:

- `--root`: Dataset root directory (default: `dataset`)
- `--sample_per_class`: Number of sample images per class (default: 12)
- `--out`: Output JSON file path (default: `dataset_summary.json`)
- `--make_sample_grid`: Generate thumbnail grids for visual inspection

**Output**:

- Corruption detection
- Image size distribution
- Class balance statistics
- Sample thumbnail grids

## 🔧 Configuration

### Backend Environment Variables

| Variable         | Description                                     | Default                      |
| ---------------- | ----------------------------------------------- | ---------------------------- |
| `DEVICE`         | PyTorch device (`cuda` or `cpu`)                | Auto-detect                  |
| `LOAD_MODELS`    | Models to load (`all` or comma-separated names) | `all`                        |
| `GEMINI_API_KEY` | Google Gemini API key for explanations          | Required for AI explanations |

### Frontend Environment Variables

| Variable        | Description          | Default                 |
| --------------- | -------------------- | ----------------------- |
| `VITE_API_BASE` | Backend API base URL | `http://localhost:8000` |

## 📦 Dependencies

### Backend

- **Deep Learning**: PyTorch 2.0+, TorchVision 0.15+
- **Web Framework**: FastAPI 0.100+, Uvicorn 0.23+
- **Computer Vision**: OpenCV 4.8+, Pillow 9.0+
- **Explainability**: TorchCAM 0.3+
- **AI Integration**: google-generativeai 0.3+
- **Utilities**: NumPy, Pandas, scikit-learn, python-dotenv

### Frontend

- **Framework**: React 19.1.1, React Router DOM 7.9.1
- **UI Library**: Material-UI 7.3.2, Emotion
- **Build Tool**: Vite 7.1.6
- **Icons**: MUI Icons Material 7.3.2

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Install Git LFS: `git lfs install`
3. Create a feature branch (`git checkout -b feature/AmazingFeature`)
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

**Note**: If you're contributing model files, they will automatically be tracked by Git LFS (configured in `.gitattributes`).

## 📥 Important: Git LFS for Model Files

This repository uses **Git LFS** to manage large model files efficiently. The actual model weights are stored in LFS, while only small pointer files are committed to Git.

### For Users Cloning the Repository

**Option 1: Automatic (Recommended)**

```bash
# Install Git LFS first
git lfs install

# Then clone - models download automatically
git clone <repository-url>
```

**Option 2: Manual Download**

```bash
# If you cloned without Git LFS
git lfs install
git lfs pull  # Downloads all LFS files
```

### Verify Model Downloads

```bash
# Check file sizes (should be 75-240 MB each)
ls -lh Backend/models/*.pth

# Or on Windows PowerShell
Get-ChildItem Backend/models/*.pth | Format-Table Name, @{L="Size (MB)";E={[math]::Round($_.Length/1MB, 2)}}
```

If files are only a few KB, they're pointer files - run `git lfs pull`.

### For Contributors Adding Models

Model files are automatically tracked by Git LFS (see `.gitattributes`):

```bash
# Just add and commit normally
git add Backend/models/newmodel.pth
git commit -m "Add new model"
git push  # LFS handles the upload automatically
```

See [`Backend/models/README.md`](Backend/models/README.md) for detailed LFS troubleshooting.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyTorch**: For the deep learning framework
- **TorchCAM**: For GradCAM implementation
- **Google Gemini**: For AI-powered explanations
- **Material-UI**: For the beautiful React components
- **FastAPI**: For the high-performance web framework

## 📧 Contact

For questions, issues, or suggestions, please open an issue on the GitHub repository.

## 🔮 Future Enhancements

- [ ] Video deepfake detection
- [ ] Real-time webcam analysis
- [ ] Additional model architectures (Vision Transformers, etc.)
- [ ] Fine-grained manipulation detection (face swap, expression, etc.)
- [ ] Batch processing API
- [ ] Model performance benchmarking dashboard
- [ ] Docker containerization
- [ ] Cloud deployment guides (AWS, GCP, Azure)

---

**Built with ❤️ for transparency in AI-generated content detection**
