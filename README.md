# Deepfake Detection using Explainable AI

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![React](https://img.shields.io/badge/react-19.1.1-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Material--UI](https://img.shields.io/badge/Material--UI-7.3.2-blue)

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Dataset Information](#-dataset-information)
- [Project Structure](#-project-structure)
- [Technical Stack](#-technical-stack)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Frontend Components](#-frontend-components)
- [Model Details](#-model-details)
- [Explainability Features](#-explainability-features)
- [Performance Metrics](#-performance-metrics)
- [Utilities](#-utilities)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Future Enhancements](#-future-enhancements)

## 🎯 Overview

An advanced deepfake detection system powered by an ensemble of state-of-the-art deep learning models with explainable AI capabilities. This project combines multiple CNN architectures with GradCAM visualization and AI-powered explanations to detect manipulated images with high accuracy and transparency.

**Key Highlights:**

- 🤖 **5-Model Ensemble**: Combines ResNet50, ResNet152V2, InceptionResNetV2, Xception, and EfficientNetB4
- 🔍 **Explainable AI**: GradCAM heatmaps + Google Gemini-powered natural language explanations
- 🌐 **Modern Web Interface**: React 19 + Material-UI with real-time analysis
- ⚡ **High Performance**: FastAPI backend with GPU acceleration support
- 📊 **Comprehensive Dataset**: 190,335+ verified images (real and AI-generated)
- 🎨 **Interactive Visualizations**: Real-time heatmap overlays showing model attention regions

## 🌟 Features

### Core Capabilities

#### 1. **Multi-Model Ensemble Architecture**

- **5 Powerful CNN Models** working in concert:
  - **ResNet50** (25.6M parameters) - Residual learning for deep networks
  - **ResNet152V2** (60.2M parameters) - Deeper residual architecture
  - **InceptionResNetV2** (55.8M parameters) - Multi-scale feature extraction
  - **Xception** (22.9M parameters) - Depthwise separable convolutions
  - **EfficientNetB4** (19.3M parameters) - Compound scaling optimization
- **Ensemble Prediction**: Weighted averaging of model confidences
- **Individual Model Selection**: Test specific models or compare results
- **Robust Decision Making**: Reduces individual model biases

#### 2. **Explainable AI (XAI) Framework**

- **GradCAM Heatmap Visualization**:
  - Shows which image regions influenced the model's decision
  - Color-coded attention maps (red = high attention, blue = low attention)
  - Weighted fusion of heatmaps from all ensemble models
  - Real-time overlay generation with 40% heatmap + 60% original image composition
- **AI-Generated Natural Language Explanations**:
  - Powered by Google Gemini 2.5 Flash API
  - Context-aware explanations based on prediction and heatmap
  - Human-readable justifications for each decision
  - Automatic fallback to generic explanations if API unavailable
- **Per-Model Confidence Breakdown**:
  - Individual confidence scores for each model
  - Ensemble confidence aggregation
  - Real vs Fake probability distribution
  - Transparency in decision-making process

#### 3. **Modern Web Interface**

- **Responsive React-based UI**:
  - Mobile-first design with adaptive layouts
  - Dark theme optimized for visual comfort
  - Material-UI components for professional appearance
  - Real-time state management with React hooks
- **Interactive Features**:
  - Drag-and-drop image upload
  - File browser integration
  - Live analysis progress indicators
  - Instant result visualization
  - Model selector dropdown (Ensemble + 5 individual models)
- **Rich Visual Feedback**:
  - Original image display with border indicators
  - Side-by-side heatmap comparison
  - Color-coded prediction badges (Green = Real, Red = Fake)
  - Confidence percentage displays
  - Detailed explanation cards

#### 4. **Production-Ready Backend**

- **FastAPI Framework**:
  - Async request handling for high performance
  - Auto-generated OpenAPI documentation (Swagger UI)
  - Type hints and data validation with Pydantic
  - RESTful API design
- **Advanced Features**:
  - CORS-enabled for cross-origin requests
  - GPU/CPU automatic device detection
  - Batch model loading on startup
  - Robust error handling and logging
  - Health check endpoints
  - Model status monitoring
- **Performance Optimizations**:
  - Model caching in memory
  - Efficient image preprocessing pipelines
  - Parallel GradCAM computation
  - Optimized tensor operations

#### 5. **Dataset Management**

- **Comprehensive Validation Tools**:
  - Automatic corruption detection
  - Image integrity verification
  - Size distribution analysis
  - Class balance statistics
  - Sample grid generation
- **Flexible Data Organization**:
  - Train/Validation/Test split structure
  - Real/Fake binary classification
  - Standardized 256×256 resolution
  - Zero corrupted images verified

#### 6. **Developer Experience**

- **Git LFS Integration**: Efficient handling of large model files
- **Comprehensive Documentation**: Setup guides, API docs, and troubleshooting
- **Environment Configuration**: `.env` support for easy customization
- **Logging & Debugging**: Detailed logs for monitoring and debugging
- **Code Quality**: Type hints, modular architecture, clear separation of concerns

## 🏛️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  React 19 + Vite + Material-UI                           │  │
│  │  • HomePage.jsx (Main container)                         │  │
│  │  • ModelSection.jsx (Analysis interface)                 │  │
│  │  • useImageAnalysis.js (State management hook)           │  │
│  │  • ImageUploadArea, ImageDisplay, HeatmapDisplay        │  │
│  │  • ExplanationBox, RealImageMessage, WaitingState       │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP/REST API
                            │ (JSON + Base64 Images)
┌───────────────────────────▼─────────────────────────────────────┐
│                        Backend Layer                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  FastAPI + Uvicorn                                       │  │
│  │  • main.py (API endpoints & routing)                     │  │
│  │  • service.py (Model Manager & inference logic)          │  │
│  │  • CORS middleware for cross-origin requests            │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Deep Learning Layer                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  PyTorch 2.0+ with CUDA Support                          │  │
│  │  ┌────────────┬──────────────┬─────────────┬──────────┐ │  │
│  │  │ ResNet50   │ ResNet152V2  │ Inception   │ Xception │ │  │
│  │  │ 224×224    │ 224×224      │ ResNetV2    │ 299×299  │ │  │
│  │  │ 98 MB      │ 236 MB       │ 299×299     │ 88 MB    │ │  │
│  │  │            │              │ 215 MB      │          │ │  │
│  │  └────────────┴──────────────┴─────────────┴──────────┘ │  │
│  │  ┌──────────────────────────────────────────────────────┐ │  │
│  │  │ EfficientNetB4 (380×380, 75 MB)                      │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                   Explainability Layer                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  TorchCAM (GradCAM Implementation)                       │  │
│  │  • Target layer extraction per model                     │  │
│  │  • Activation map generation                             │  │
│  │  • Weighted heatmap fusion                               │  │
│  │  • Overlay composition (OpenCV)                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Google Gemini 2.5 Flash API                             │  │
│  │  • Natural language explanation generation               │  │
│  │  • Context-aware reasoning                               │  │
│  │  • Vision-language integration                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

```
User Upload Image
      │
      ▼
┌─────────────────────┐
│ Frontend Processing │
│ • File validation   │──────┐
│ • Base64 encoding   │      │
│ • State management  │      │
└─────────────────────┘      │
      │                      │
      ▼                      │
┌─────────────────────┐      │
│ HTTP POST Request   │      │
│ /predict/           │      │
│ ?model=<name>       │      │
└─────────────────────┘      │
      │                      │
      ▼                      ▼
┌──────────────────────────────────┐
│ Backend API (FastAPI)            │
│ 1. Receive multipart/form-data   │
│ 2. Decode & open with Pillow     │
│ 3. Convert to RGB                │
└──────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────┐
│ Model Manager (service.py)       │
│ Single Model OR Ensemble?        │
└──────┬───────────────────────┬───┘
       │                       │
       ▼                       ▼
┌─────────────┐      ┌──────────────────┐
│Single Model │      │ Ensemble (5)     │
│Prediction   │      │ • Parallel proc  │
└─────────────┘      │ • Prob averaging │
       │             │ • Weighted CAMs  │
       │             └──────────────────┘
       │                       │
       └───────────┬───────────┘
                   ▼
       ┌────────────────────┐
       │ For Each Model:    │
       │ 1. Preprocess      │
       │    (resize, norm)  │
       │ 2. Forward pass    │
       │ 3. Softmax/Sigmoid │
       │ 4. GradCAM compute │
       └────────────────────┘
                   │
                   ▼
       ┌────────────────────┐
       │ Aggregate Results  │
       │ • Predictions      │
       │ • Confidences      │
       │ • Fuse CAMs        │
       └────────────────────┘
                   │
                   ▼
       ┌────────────────────┐
       │ Generate Heatmap   │
       │ 1. Normalize CAM   │
       │ 2. Apply colormap  │
       │ 3. Blend overlay   │
       │ 4. Encode base64   │
       └────────────────────┘
                   │
                   ▼
       ┌────────────────────┐
       │ Call Gemini API    │
       │ (if Fake detected) │
       │ • Send heatmap     │
       │ • Get explanation  │
       └────────────────────┘
                   │
                   ▼
       ┌────────────────────┐
       │ JSON Response      │
       │ {                  │
       │   prediction,      │
       │   confidence,      │
       │   probabilities,   │
       │   heatmap (base64),│
       │   explanation,     │
       │   per_model_data   │
       │ }                  │
       └────────────────────┘
                   │
                   ▼
       ┌────────────────────┐
       │ Frontend Render    │
       │ • Display results  │
       │ • Show heatmap     │
       │ • Show explanation │
       │ • Update UI state  │
       └────────────────────┘
```

### Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     Frontend Components                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  App.jsx (Router)                                            │
│    │                                                          │
│    └──► HomePage.jsx (Theme Provider + Dark Mode)           │
│           │                                                   │
│           ├──► Navbar.jsx (GitHub link)                      │
│           │                                                   │
│           └──► ModelSection.jsx (Main Interface)             │
│                  │                                            │
│                  ├──► useImageAnalysis.js (Hook)             │
│                  │      • Image state management             │
│                  │      • API communication                  │
│                  │      • Model selection                    │
│                  │      • Analysis orchestration             │
│                  │                                            │
│                  ├──► ImageUploadArea.jsx                    │
│                  │      • Drag & drop handling               │
│                  │      • File picker trigger                │
│                  │      • File validation                    │
│                  │                                            │
│                  ├──► ImageDisplay.jsx                       │
│                  │      • Original image preview             │
│                  │      • Reset functionality                │
│                  │                                            │
│                  ├──► HeatmapDisplay.jsx                     │
│                  │      • Heatmap visualization              │
│                  │      • Prediction badges                  │
│                  │      • Confidence display                 │
│                  │      • Loading states                     │
│                  │                                            │
│                  ├──► ExplanationBox.jsx                     │
│                  │      • AI explanation rendering           │
│                  │      • Color-coded by prediction          │
│                  │      • Error handling display             │
│                  │                                            │
│                  ├──► RealImageMessage.jsx                   │
│                  │      • Success state for real images      │
│                  │                                            │
│                  └──► WaitingState.jsx                       │
│                         • Analysis progress indicator        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

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
  - InceptionResNetV2: 299×299
  - Xception: 299×299
  - EfficientNetB4: 380×380

- **Output**: 2-class softmax (Real vs Fake)

- **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Model Performance

| Model                 | Train Acc | Train Loss | Val Acc    | Val Loss |
| --------------------- | --------- | ---------- | ---------- | -------- |
| **EfficientNetB4**    | 99.82%    | 0.0050     | 98.68%     | 0.0787   |
| **InceptionResNetV2** | 99.62%    | 0.0093     | **98.88%** | 0.0441   |
| **ResNet152V2**       | 99.72%    | 0.0085     | 98.87%     | 0.0506   |
| **Xception**          | 99.41%    | 0.0048     | 98.79%     | 0.0658   |
| **ResNet50**          | 99.18%    | 0.0206     | 98.57%     | 0.0438   |

**Key Metrics**:

- Average validation accuracy: **98.76%**
- Best validation accuracy: **98.88%** (InceptionResNetV2)
- Minimal overfitting: avg generalization gap of 0.79%
- All models achieve >98.5% validation accuracy

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
