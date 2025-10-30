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
- [Backend Architecture](#-backend-architecture)
- [Model Details](#-model-details)
- [Explainability Features](#-explainability-features)
- [Performance Metrics](#-performance-metrics)
- [Utilities & Tools](#-utilities--tools)
- [Testing & Validation](#-testing--validation)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Future Enhancements](#-future-enhancements)
- [Contact](#-contact)

## 🎯 Overview

An **enterprise-grade deepfake detection system** powered by an ensemble of state-of-the-art deep learning models with comprehensive explainable AI capabilities. This project combines multiple CNN architectures with GradCAM visualization and AI-powered natural language explanations to detect manipulated images with high accuracy, transparency, and user trust.

**Key Highlights:**

- 🤖 **5-Model Ensemble Architecture**: Combines ResNet50, ResNet152V2, InceptionResNetV2, Xception, and EfficientNetB4
- 🔍 **Advanced Explainable AI**: GradCAM heatmaps + Google Gemini 2.5 Flash-powered natural language explanations
- 🌐 **Modern Web Interface**: React 19 + Material-UI with responsive design and real-time analysis
- ⚡ **High Performance Backend**: FastAPI with async support and GPU acceleration
- 📊 **Massive Dataset**: 190,335+ verified images (real and AI-generated)
- 🎨 **Interactive Visualizations**: Real-time heatmap overlays showing model attention regions
- 📦 **Production Ready**: Docker support, comprehensive logging, error handling, and monitoring
- 🔒 **Secure & Scalable**: CORS configuration, environment-based secrets, scalable architecture

## 🌟 Features

### Core Capabilities

#### 1. **Multi-Model Ensemble Architecture**

- **5 Powerful CNN Models** working in concert:
  - **ResNet50** (25.6M parameters) - Residual learning for deep networks, 224×224 input
  - **ResNet152V2** (60.2M parameters) - Deeper residual architecture, 224×224 input
  - **InceptionResNetV2** (55.8M parameters) - Multi-scale feature extraction, 299×299 input
  - **Xception** (22.9M parameters) - Depthwise separable convolutions, 299×299 input
  - **EfficientNetB4** (19.3M parameters) - Compound scaling optimization, 380×380 input
- **Ensemble Prediction**: Weighted averaging of softmax probabilities across all models
- **Individual Model Selection**: Test specific models independently or compare results
- **Robust Decision Making**: Reduces individual model biases through diversity
- **Dynamic Model Loading**: Selective model loading based on configuration

#### 2. **Explainable AI (XAI) Framework**

- **GradCAM Heatmap Visualization**:
  - Gradient-weighted Class Activation Mapping for visual explanations
  - Shows which image regions influenced the model's decision
  - Color-coded attention maps (red/yellow = high attention, blue = low attention)
  - Weighted fusion of heatmaps from all ensemble models
  - Real-time overlay generation with 40% heatmap + 60% original image composition
  - Model-specific target layers for optimal activation extraction
- **AI-Generated Natural Language Explanations**:
  - Powered by Google Gemini 2.5 Flash multimodal API
  - Context-aware explanations based on prediction label and heatmap
  - Human-readable justifications for each decision in 2-3 sentences
  - Automatic fallback to generic explanations if API unavailable
  - Vision-language integration for deeper understanding
- **Per-Model Confidence Breakdown**:
  - Individual confidence scores for each model
  - Ensemble confidence aggregation
  - Real vs Fake probability distribution
  - Complete transparency in decision-making process
  - JSON-formatted detailed metrics

#### 3. **Modern Web Interface**

- **Responsive React-based UI**:
  - Mobile-first design with adaptive layouts for all screen sizes
  - Dark theme optimized for visual comfort and professional appearance
  - Material-UI 7.3.2 components for consistent, polished interface
  - Real-time state management with React hooks (useImageAnalysis)
  - Smooth animations and transitions for better UX
- **Interactive Features**:
  - Drag-and-drop image upload with visual feedback
  - File browser integration with image format validation
  - Live analysis progress indicators with loading states
  - Instant result visualization without page reloads
  - Model selector dropdown (Ensemble + 5 individual models)
  - Reset functionality to analyze multiple images
- **Rich Visual Feedback**:
  - Original image display with responsive borders
  - Side-by-side heatmap comparison layout
  - Color-coded prediction badges (Green = Real, Red = Fake, Orange = Error)
  - Confidence percentage displays with precision
  - Detailed explanation cards with contextual styling
  - Smooth state transitions and animations

#### 4. **Production-Ready Backend**

- **FastAPI Framework**:
  - Async/await request handling for high concurrent performance
  - Auto-generated OpenAPI documentation (Swagger UI at /docs)
  - Type hints and automatic data validation with Pydantic
  - RESTful API design following best practices
  - Built-in request/response schemas
- **Advanced Features**:
  - CORS middleware for secure cross-origin requests
  - GPU/CPU automatic device detection with fallback
  - Batch model loading on application startup
  - Comprehensive error handling with proper HTTP status codes
  - Structured logging with log levels (INFO, WARNING, ERROR)
  - Health check and status monitoring endpoints
  - Model availability checking before inference
- **Performance Optimizations**:
  - Model caching in memory (singleton pattern)
  - Efficient image preprocessing pipelines with transforms
  - Parallel GradCAM computation across models
  - Optimized tensor operations with PyTorch JIT (where applicable)
  - Minimal memory footprint with cleanup
  - Fast multipart/form-data parsing

#### 5. **Dataset Management**

- **Comprehensive Validation Tools** (`check_dataset.py`):
  - Automatic image corruption detection with PIL verification
  - Image integrity verification (file header checks)
  - Size distribution analysis across splits
  - Class balance statistics and imbalance detection
  - Sample grid generation for visual inspection
  - JSON summary export for documentation
  - Progress bars with tqdm for large datasets
- **Flexible Data Organization**:
  - Train/Validation/Test split structure (73.5% / 20.7% / 5.7%)
  - Real/Fake binary classification folders
  - Standardized 256×256 resolution across all images
  - Zero corrupted images verified (100% integrity)
  - Hierarchical directory structure for scalability

#### 6. **Developer Experience**

- **Git LFS Integration**:
  - Efficient handling of large model files (~1.66 GB total)
  - Automatic tracking for `.pth`, `.pt`, `.h5`, `.onnx`, `.pkl` files
  - Pointer files in repo, actual files in LFS storage
  - Simple clone-and-go workflow
- **Comprehensive Documentation**:
  - README.md with complete project overview
  - SETUP_GUIDE.md with step-by-step installation
  - QUICK_REFERENCE.md for common commands
  - Backend/DATASET.md for dataset download
  - Backend/models/README.md for model files
  - GIT_LFS_IMPLEMENTATION.md for LFS details
  - Inline code comments and docstrings
- **Environment Configuration**:
  - `.env` support for secrets and configuration
  - Environment variable validation
  - Separate configs for dev/prod environments
  - Configurable model loading (all or selective)
- **Logging & Debugging**:
  - Detailed logs for monitoring and debugging
  - Configurable log levels
  - Request/response logging
  - Error stack traces
  - Performance metrics tracking
- **Code Quality**:
  - Type hints throughout codebase
  - Modular architecture with clear separation
  - Single Responsibility Principle
  - DRY (Don't Repeat Yourself) practices
  - Consistent naming conventions

## 🏛️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  React 19 + Vite + Material-UI                           │  │
│  │  Port: 5173 (Development) / 4173 (Preview)               │  │
│  │  ────────────────────────────────────────────────────    │  │
│  │  • App.jsx (Router + Root Component)                     │  │
│  │  • HomePage.jsx (Theme Provider + Dark Mode)             │  │
│  │  • Navbar.jsx (GitHub Navigation)                        │  │
│  │  • ModelSection.jsx (Main Analysis Interface)            │  │
│  │  • useImageAnalysis.js (Custom Hook - State Management)  │  │
│  │                                                           │  │
│  │  Sub-Components:                                         │  │
│  │  • ImageUploadArea.jsx (Drag & Drop + File Picker)       │  │
│  │  • ImageDisplay.jsx (Original Image Display)             │  │
│  │  • HeatmapDisplay.jsx (GradCAM Visualization)            │  │
│  │  • ExplanationBox.jsx (AI Explanation Rendering)         │  │
│  │  • RealImageMessage.jsx (Success State Display)          │  │
│  │  • WaitingState.jsx (Loading Indicators)                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP/REST API (JSON + Base64)
                            │ CORS-enabled, async requests
┌───────────────────────────▼─────────────────────────────────────┐
│                        Backend Layer                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  FastAPI + Uvicorn ASGI Server                           │  │
│  │  Port: 8000 (Configurable)                               │  │
│  │  ────────────────────────────────────────────────────    │  │
│  │  • main.py (API Endpoints, Routing, Middleware)          │  │
│  │    - GET  /            (Health Check)                    │  │
│  │    - GET  /models/     (List Loaded Models)              │  │
│  │    - GET  /status/     (System Status & Device Info)     │  │
│  │    - POST /predict/    (Image Analysis Endpoint)         │  │
│  │                                                           │  │
│  │  • service.py (Core Logic)                               │  │
│  │    - ModelManager Class (Singleton)                      │  │
│  │    - Model Loading & Checkpoint Management               │  │
│  │    - Inference Orchestration                             │  │
│  │    - GradCAM Generation                                  │  │
│  │    - Gemini API Integration                              │  │
│  │                                                           │  │
│  │  • CORS Middleware (Cross-Origin Security)               │  │
│  │  • Error Handlers & Exception Logging                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Deep Learning Layer                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  PyTorch 2.0+ with CUDA/CPU Support                      │  │
│  │  Device: Auto-detect (CUDA if available, else CPU)       │  │
│  │  ────────────────────────────────────────────────────    │  │
│  │                                                           │  │
│  │  ┌────────────┬──────────────┬─────────────┬──────────┐ │  │
│  │  │ ResNet50   │ ResNet152V2  │ Inception   │ Xception │ │  │
│  │  │            │              │ ResNetV2    │          │ │  │
│  │  │ Input:     │ Input:       │ Input:      │ Input:   │ │  │
│  │  │ 224×224    │ 224×224      │ 299×299     │ 299×299  │ │  │
│  │  │            │              │             │          │ │  │
│  │  │ Size:      │ Size:        │ Size:       │ Size:    │ │  │
│  │  │ ~98 MB     │ ~236 MB      │ ~215 MB     │ ~88 MB   │ │  │
│  │  │            │              │             │          │ │  │
│  │  │ Params:    │ Params:      │ Params:     │ Params:  │ │  │
│  │  │ 25.6M      │ 60.2M        │ 55.8M       │ 22.9M    │ │  │
│  │  │            │              │             │          │ │  │
│  │  │ Target:    │ Target:      │ Target:     │ Target:  │ │  │
│  │  │ layer4[-1] │ layer4[-1]   │ Mixed_7c    │ block4   │ │  │
│  │  │ .conv2     │ .conv2       │ (conv2d_7b) │          │ │  │
│  │  └────────────┴──────────────┴─────────────┴──────────┘ │  │
│  │                                                           │  │
│  │  ┌──────────────────────────────────────────────────────┐ │  │
│  │  │ EfficientNetB4                                       │ │  │
│  │  │ Input: 380×380 | Size: ~75 MB | Params: 19.3M       │ │  │
│  │  │ Target: blocks[-1][-1].conv_pwl                      │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  │                                                           │  │
│  │  Preprocessing Pipeline:                                 │  │
│  │  • Resize to model-specific dimensions                   │  │
│  │  • Convert to Tensor                                     │  │
│  │  • Normalize with ImageNet stats                         │  │
│  │    Mean: [0.485, 0.456, 0.406]                           │  │
│  │    Std:  [0.229, 0.224, 0.225]                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                   Explainability Layer                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  TorchCAM (GradCAM Implementation)                       │  │
│  │  ────────────────────────────────────────────────────    │  │
│  │  • Target layer extraction per model architecture        │  │
│  │  • Gradient computation with backward pass               │  │
│  │  • Activation map generation (class-specific)            │  │
│  │  • CAM normalization and resizing to original size       │  │
│  │  • Weighted heatmap fusion (confidence-based weights)    │  │
│  │  • Overlay composition with OpenCV:                      │  │
│  │    - Apply JET colormap to normalized CAM                │  │
│  │    - Blend: 40% heatmap + 60% original image             │  │
│  │    - Encode to base64 PNG for transmission               │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Google Gemini 2.5 Flash API                             │  │
│  │  ────────────────────────────────────────────────────    │  │
│  │  • Natural language explanation generation               │  │
│  │  • Context-aware reasoning based on prediction label     │  │
│  │  • Vision-language multimodal integration                │  │
│  │  • Input: Prediction label + Heatmap overlay (base64)    │  │
│  │  • Output: 2-3 sentence human-readable explanation       │  │
│  │  • Fallback: Generic explanation if API fails            │  │
│  │  • Rate limiting & error handling                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

```
┌─────────────────┐
│ User Action     │
│ Upload Image    │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────┐
│ Frontend Processing          │
│ ────────────────────────     │
│ 1. File validation           │
│    • Type check (image/*)    │
│    • Size limit check        │
│ 2. FileReader API            │
│    • Convert to base64       │
│    • Generate preview URL    │
│ 3. State Management          │
│    • Update selectedImage    │
│    • Set isAnalyzing = true  │
│    • Clear previous results  │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ HTTP POST Request            │
│ ────────────────────────     │
│ URL: /predict/               │
│ Query: ?model=<name>         │
│        (optional)            │
│ Body: multipart/form-data    │
│   file: <binary image data>  │
│ Headers:                     │
│   Content-Type:              │
│     multipart/form-data      │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Backend API Endpoint                 │
│ ────────────────────────────────     │
│ @app.post("/predict/")               │
│ 1. Receive UploadFile                │
│ 2. Read file bytes: await file.read()│
│ 3. Open with PIL: Image.open()       │
│ 4. Convert to RGB: .convert("RGB")   │
│ 5. Extract model param from query    │
└──────────┬───────────────────────────┘
           │
           ▼
┌───────────────────────────────────────┐
│ Model Manager Dispatcher              │
│ ─────────────────────────────────     │
│ Route based on model parameter:       │
│                                       │
│ if model specified:                   │
│   → predict_single(model, image)      │
│ else:                                 │
│   → predict_ensemble(image)           │
└───────┬───────────────────────────────┘
        │
        ├─────────────────┬────────────────┐
        │                 │                │
        ▼                 ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│Single Model  │  │ Ensemble     │  │ Ensemble     │
│Prediction    │  │ Model 1      │  │ Model 2-5    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                │
       │                 └────────┬───────┘
       │                          │
       └──────────────┬───────────┘
                      ▼
┌─────────────────────────────────────────┐
│ For Each Model (Parallel in Ensemble):  │
│ ────────────────────────────────────     │
│ 1. Get model-specific preprocessor      │
│    • Resize to input dimensions          │
│    • ToTensor transformation             │
│    • Normalize with ImageNet stats       │
│                                          │
│ 2. Prepare input tensor                 │
│    • Add batch dimension: unsqueeze(0)   │
│    • Move to device: .to(device)         │
│    • Enable gradients: requires_grad_()  │
│                                          │
│ 3. Forward pass through model            │
│    • output = model(input_tensor)        │
│                                          │
│ 4. Compute probabilities                │
│    • For Xception: direct sigmoid output │
│    • For others: F.softmax(output, dim=1)│
│                                          │
│ 5. Handle class inversion                │
│    • InceptionResNet & EfficientNet:     │
│      Index 0 = Real, Index 1 = Fake      │
│    • Others:                             │
│      Index 0 = Fake, Index 1 = Real      │
│    • Unify to [Fake, Real] format        │
│                                          │
│ 6. Determine prediction                 │
│    • pred_idx = argmax(probabilities)    │
│    • label = "Real" if pred_idx==1       │
│              else "Fake"                 │
│                                          │
│ 7. Extract GradCAM                       │
│    • cam_extractor = GradCAM(model)      │
│    • Determine target class index        │
│      (handle model-specific inversions)  │
│    • cam = cam_extractor(class_idx, out) │
│    • Normalize CAM to [0, 1]             │
│    • Resize to original image size       │
└─────────────┬───────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────┐
│ Aggregate Results (Ensemble Only)        │
│ ─────────────────────────────────────    │
│ 1. Collect all unified probabilities     │
│    • Stack: [[Fake, Real], ...]          │
│                                          │
│ 2. Average probabilities                 │
│    • ensemble_probs = mean(all_probs)    │
│                                          │
│ 3. Determine ensemble prediction         │
│    • pred_idx = argmax(ensemble_probs)   │
│    • label = "Real" if pred_idx==1       │
│              else "Fake"                 │
│                                          │
│ 4. Fuse GradCAMs                         │
│    • weights = confidence per model      │
│    • Normalize weights: sum = 1.0        │
│    • Weighted sum of CAMs                │
│    • ensemble_cam = Σ(weight_i * cam_i)  │
│    • Normalize final CAM                 │
└─────────────┬────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────┐
│ Generate Heatmap Overlay                 │
│ ─────────────────────────────────────    │
│ 1. Denormalize input image               │
│    • Reverse ImageNet normalization      │
│    • Clip to [0, 1] range                │
│    • Resize to original dimensions       │
│                                          │
│ 2. Apply colormap to CAM                 │
│    • Convert CAM to uint8: cam * 255     │
│    • Apply JET colormap (OpenCV)         │
│    • Convert BGR to RGB                  │
│                                          │
│ 3. Create overlay                        │
│    • Blend: 0.4 * heatmap + 0.6 * image  │
│    • Clip to [0, 1]                      │
│    • Convert to uint8: * 255             │
│                                          │
│ 4. Encode to base64                      │
│    • imencode to PNG format              │
│    • base64 encode buffer                │
│    • Decode to UTF-8 string              │
└─────────────┬────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────┐
│ Generate AI Explanation (if Fake)        │
│ ─────────────────────────────────────    │
│ if prediction == "Fake":                 │
│   1. Check Gemini API availability       │
│   2. Prepare prompt:                     │
│      "Model predicted {label}.           │
│       Heatmap shows attention regions.   │
│       Explain why in 2-3 sentences."     │
│   3. Call Gemini API:                    │
│      • Model: gemini-2.5-flash           │
│      • Input: [prompt, heatmap_base64]   │
│   4. Extract explanation text            │
│   5. Fallback on error:                  │
│      Generic message                     │
│ else:                                    │
│   Generic "Real" explanation             │
└─────────────┬────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────┐
│ Construct JSON Response                  │
│ ─────────────────────────────────────    │
│ Single Model Response:                   │
│ {                                        │
│   "model": "resnet50",                   │
│   "prediction": "Fake",                  │
│   "confidence": 0.9532,                  │
│   "probabilities": {                     │
│     "real": 0.0468,                      │
│     "fake": 0.9532                       │
│   },                                     │
│   "heatmap": "data:image/png;base64,...",│
│   "explanation": "The model detected..." │
│ }                                        │
│                                          │
│ Ensemble Response:                       │
│ {                                        │
│   "ensemble_prediction": "Fake",         │
│   "ensemble_confidence": 0.9621,         │
│   "ensemble_probabilities": {            │
│     "real": 0.0379,                      │
│     "fake": 0.9621                       │
│   },                                     │
│   "per_model_confidences": {             │
│     "resnet50": {"real": 0.05, ...},     │
│     "resnet152": {"real": 0.03, ...},    │
│     ...                                  │
│   },                                     │
│   "heatmap": "data:image/png;base64,...",│
│   "explanation": "The ensemble..."       │
│ }                                        │
└─────────────┬────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────┐
│ Frontend State Update                    │
│ ─────────────────────────────────────    │
│ 1. Parse JSON response                   │
│ 2. Extract data:                         │
│    • prediction = response.prediction    │
│      or response.ensemble_prediction     │
│    • confidence = response.confidence    │
│      or response.ensemble_confidence     │
│    • heatmap = response.heatmap          │
│    • explanation = response.explanation  │
│ 3. Update React state:                   │
│    • setPrediction(prediction)           │
│    • setConfidence(confidence)           │
│    • setHeatmapUrl(heatmap)              │
│    • setExplanation(explanation)         │
│    • setIsAnalyzing(false)               │
└─────────────┬────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────┐
│ UI Rendering                             │
│ ─────────────────────────────────────    │
│ • ImageDisplay: Show original image      │
│ • HeatmapDisplay: Render heatmap overlay │
│   - Color-coded badge (Red/Green)        │
│   - Confidence percentage                │
│ • ExplanationBox: Display AI explanation │
│   - Contextual styling based on result   │
│ • Animation: Smooth transitions          │
└──────────────────────────────────────────┘
```

## 📊 Dataset Information

### Dataset Overview

The models are trained on a comprehensive, large-scale dataset of real and AI-generated face images sourced from Kaggle, specifically curated for deepfake detection research.

**Dataset Source**: [140k Real and Fake Faces - Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

### Comprehensive Dataset Statistics

| Split      | Fake Images | Real Images | Total       | Percentage | Avg per Class |
| ---------- | ----------- | ----------- | ----------- | ---------- | ------------- |
| Training   | 70,001      | 70,001      | 140,002     | 73.54%     | 70,001        |
| Validation | 19,641      | 19,787      | 39,428      | 20.71%     | 19,714        |
| Test       | 5,492       | 5,413       | 10,905      | 5.73%      | 5,452.5       |
| **Total**  | **95,134**  | **95,201**  | **190,335** | **100%**   | **95,167.5**  |

**Class Balance**: 99.97% balanced (95,134 fake vs 95,201 real = 49.98% vs 50.02%)

### Technical Specifications

| Specification        | Value             | Description                    |
| -------------------- | ----------------- | ------------------------------ |
| **Image Resolution** | 256×256 pixels    | Standardized across all images |
| **Color Space**      | RGB (3 channels)  | Full color information         |
| **Bit Depth**        | 8-bit per channel | Standard image encoding        |
| **File Format**      | JPEG              | Compressed format              |
| **Compression**      | Variable quality  | Optimized file size            |
| **Total Storage**    | ~5.2 GB           | Complete dataset size          |
| **Image Integrity**  | 100% verified     | Zero corrupted files           |
| **Missing Files**    | 0                 | Complete dataset               |
| **Duplicate Files**  | 0                 | All unique images              |

### Dataset Structure

```
dataset/
├── Train/                           # Training Set (73.54%)
│   ├── Fake/                        # 70,001 AI-generated faces
│   │   ├── fake_0.jpg               # StyleGAN generated
│   │   ├── fake_1.jpg               # ProGAN generated
│   │   ├── fake_2.jpg               # Various GAN architectures
│   │   ├── ...
│   │   └── fake_70000.jpg
│   └── Real/                        # 70,001 authentic faces
│       ├── real_0.jpg               # FFHQ dataset
│       ├── real_1.jpg               # CelebA dataset
│       ├── real_2.jpg               # Various sources
│       ├── ...
│       └── real_70000.jpg
│
├── Validation/                      # Validation Set (20.71%)
│   ├── Fake/                        # 19,641 AI-generated faces
│   │   ├── fake_0.jpg
│   │   ├── ...
│   │   └── fake_19640.jpg
│   └── Real/                        # 19,787 authentic faces
│       ├── real_0.jpg
│       ├── ...
│       └── real_19786.jpg
│
└── Test/                            # Test Set (5.73%)
    ├── Fake/                        # 5,492 AI-generated faces
    │   ├── fake_0.jpg
    │   ├── ...
    │   └── fake_5491.jpg
    └── Real/                        # 5,413 authentic faces
        ├── real_0.jpg
        ├── ...
        └── real_5412.jpg
```

### Dataset Verification Results

From automated validation with `check_dataset.py`:

```json
{
  "root": "dataset",
  "splits": {
    "train": {
      "counts": {
        "Fake": 70001,
        "Real": 70001
      },
      "corrupt": [],
      "sizes": {
        "256x256": 140002
      },
      "samples": [...],
      "sample_grid": "dataset/sample_grid_train.jpg"
    },
    "validation": {
      "counts": {
        "Fake": 19641,
        "Real": 19787
      },
      "corrupt": [],
      "sizes": {
        "256x256": 39428
      },
      "samples": [...],
      "sample_grid": "dataset/sample_grid_validation.jpg"
    },
    "test": {
      "counts": {
        "Fake": 5492,
        "Real": 5413
      },
      "corrupt": [],
      "sizes": {
        "256x256": 10905
      },
      "samples": [...],
      "sample_grid": "dataset/sample_grid_test.jpg"
    }
  }
}
```

### Dataset Characteristics

#### Fake (AI-Generated) Images:

- **Generation Methods**:
  - StyleGAN (NVIDIA)
  - Progressive GAN (ProGAN)
  - Other state-of-the-art GAN architectures
- **Artifacts**:
  - Inconsistent facial features
  - Unnatural hair textures
  - Background artifacts
  - Pupil irregularities
  - Teeth abnormalities
  - Asymmetric facial structures
- **Diversity**: Multiple GAN architectures ensure varied artifact patterns
- **Quality**: High-resolution, realistic-looking faces to challenge detection

#### Real (Authentic) Images:

- **Sources**:
  - FFHQ (Flickr-Faces-HQ) dataset
  - CelebA dataset
  - Other curated face datasets
- **Characteristics**:
  - Natural photographs of real people
  - Diverse demographics (age, ethnicity, gender)
  - Various lighting conditions
  - Different angles and expressions
  - Real-world photography artifacts (blur, noise, compression)
- **Quality**: Professional and amateur photography
- **Diversity**: Wide range of real-world conditions

### Dataset Access

⚠️ **IMPORTANT**: Dataset is **NOT included** in this repository due to size constraints (~5.2 GB).

📥 **Download Instructions**:

1. **Automated Download** (Recommended):

   ```bash
   cd Backend
   pip install kaggle
   # Configure Kaggle API credentials (see Backend/DATASET.md)
   kaggle datasets download -d xhlulu/140k-real-and-fake-faces
   unzip 140k-real-and-fake-faces.zip -d dataset/
   ```

2. **Manual Download**:
   - Visit [Kaggle Dataset Page](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
   - Download ZIP file
   - Extract to `Backend/dataset/`

📘 **Detailed Guide**: See [`Backend/DATASET.md`](Backend/DATASET.md) for:

- Kaggle API setup instructions
- Download verification steps
- Alternative download methods
- Troubleshooting tips

🔍 **Dataset Validation**:
After download, verify integrity:

```bash
cd Backend
python check_dataset.py --root dataset --out dataset_summary.json --make_sample_grid
```

## 🗂️ Project Structure

### Complete Directory Tree

```
F:\Python\ML Project/
├── .gitattributes              # Git LFS tracking configuration
├── .gitignore                  # Git ignore patterns (Python, Node, etc.)
├── README.md                   # 📘 This comprehensive documentation
├── README_COMPREHENSIVE.md     # 📚 Extended documentation (this file)
├── SETUP_GUIDE.md              # 📖 Step-by-step installation guide
├── QUICK_REFERENCE.md          # ⚡ Quick command reference
├── GIT_LFS_IMPLEMENTATION.md   # 📦 Git LFS setup and usage guide
│
├── Backend/                    # 🐍 Python FastAPI backend service
│   ├── main.py                 # FastAPI app, endpoints, middleware
│   ├── service.py              # Model Manager, inference, GradCAM
│   ├── check_dataset.py        # Dataset validation utility
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # Environment variables (not in repo)
│   ├── .gitignore              # Backend-specific git ignores
│   ├── DATASET.md              # 📥 Dataset download instructions
│   ├── dataset_summary.json    # Generated dataset statistics
│   │
│   ├── models/                 # 🧠 Pre-trained model weights (Git LFS)
│   │   ├── __init__.py         # Python package marker
│   │   ├── xception.py         # Custom Xception implementation
│   │   ├── README.md           # 📦 Model files & LFS guide
│   │   │
│   │   ├── resnet50.pth        # 📦 ResNet50 weights (98 MB, LFS)
│   │   ├── resnet152v2.pth     # 📦 ResNet152V2 weights (236 MB, LFS)
│   │   ├── inceptionresnetv2.pth # 📦 InceptionResNetV2 (215 MB, LFS)
│   │   ├── xception.pth        # 📦 Xception weights (88 MB, LFS)
│   │   ├── efficientnetb4.pth  # 📦 EfficientNetB4 (75 MB, LFS)
│   │   │
│   │   └── __pycache__/        # Compiled Python files
│   │
│   ├── dataset/                # 📊 Training data (NOT in repo)
│   │   │                       # Download from Kaggle (see DATASET.md)
│   │   ├── Train/
│   │   │   ├── Fake/           # 70,001 fake images (256×256)
│   │   │   └── Real/           # 70,001 real images (256×256)
│   │   ├── Validation/
│   │   │   ├── Fake/           # 19,641 fake images
│   │   │   └── Real/           # 19,787 real images
│   │   └── Test/
│   │       ├── Fake/           # 5,492 fake images
│   │       └── Real/           # 5,413 real images
│   │
│   └── __pycache__/            # Compiled Python files
│
└── Frontend/                   # ⚛️ React + Vite frontend application
    ├── index.html              # HTML entry point
    ├── package.json            # NPM dependencies & scripts
    ├── package-lock.json       # Locked dependency versions
    ├── vite.config.js          # Vite build configuration
    ├── eslint.config.js        # ESLint code quality rules
    ├── .env                    # Environment variables (not in repo)
    ├── README.md               # Frontend-specific documentation
    │
    ├── public/                 # Static assets served directly
    │   └── (vite.svg, etc.)
    │
    ├── src/                    # ⚛️ React source code
    │   ├── main.jsx            # React app entry point
    │   ├── App.jsx             # Root component with routing
    │   ├── index.css           # Global styles
    │   │
    │   ├── assets/             # Images, icons, media files
    │   │
    │   └── components/         # React components
    │       │
    │       ├── HomePage.jsx    # Main page container
    │       │                   # - Theme provider (dark mode)
    │       │                   # - Layout orchestration
    │       │
    │       ├── Navbar/
    │       │   └── Navbar.jsx  # Navigation bar
    │       │                   # - GitHub repository link
    │       │                   # - Sticky positioning
    │       │
    │       └── ModelSection/   # 🎯 Main analysis interface
    │           ├── index.js    # Component exports
    │           ├── ModelSection.jsx      # Main container
    │           │                         # - Model selector
    │           │                         # - Layout management
    │           │
    │           ├── ImageUploadArea.jsx   # Upload interface
    │           │                         # - Drag & drop
    │           │                         # - File picker
    │           │
    │           ├── ImageDisplay.jsx      # Original image
    │           │                         # - Preview display
    │           │                         # - Reset button
    │           │
    │           ├── HeatmapDisplay.jsx    # Heatmap visualization
    │           │                         # - GradCAM overlay
    │           │                         # - Prediction badge
    │           │                         # - Confidence display
    │           │
    │           ├── ExplanationBox.jsx    # AI explanation
    │           │                         # - Color-coded styling
    │           │                         # - Explanation text
    │           │
    │           ├── RealImageMessage.jsx  # Success message
    │           │                         # - Real image feedback
    │           │
    │           ├── WaitingState.jsx      # Loading state
    │           │                         # - Analysis progress
    │           │
    │           └── hooks/
    │               └── useImageAnalysis.js  # Custom React hook
    │                                        # - State management
    │                                        # - API communication
    │                                        # - File handling
    │
    ├── dist/                   # Production build output (generated)
    └── node_modules/           # NPM packages (generated, not in repo)
```

### File Count Summary

| Category                 | Count   | Description                                        |
| ------------------------ | ------- | -------------------------------------------------- |
| **Python Files**         | 4       | main.py, service.py, check_dataset.py, xception.py |
| **JavaScript/JSX Files** | 13      | React components and configuration                 |
| **Model Files**          | 5       | Pre-trained weights (Git LFS)                      |
| **Configuration Files**  | 8       | package.json, vite.config.js, .env, etc.           |
| **Documentation Files**  | 8       | README, guides, references                         |
| **Dataset Images**       | 190,335 | Not in repository, download separately             |

### Key Directories

#### Backend/

- **Purpose**: Python-based API server for model inference
- **Framework**: FastAPI + Uvicorn ASGI
- **Key Files**:
  - `main.py`: API endpoints, routing, CORS middleware
  - `service.py`: Core ML logic, model management, GradCAM
  - `check_dataset.py`: Dataset validation and statistics
- **Dependencies**: Listed in `requirements.txt`

#### Backend/models/

- **Purpose**: Pre-trained deep learning model weights
- **Storage**: Git LFS (Large File Storage)
- **Total Size**: ~712 MB (combined)
- **Tracked Files**: All `.pth` files automatically tracked
- **Note**: Requires Git LFS installation to download

#### Backend/dataset/

- **Purpose**: Training, validation, and test images
- **Status**: NOT included in repository
- **Download**: Required from Kaggle (see DATASET.md)
- **Size**: ~5.2 GB total
- **Structure**: Train/Val/Test splits with Fake/Real classes

#### Frontend/

- **Purpose**: User interface for image analysis
- **Framework**: React 19 + Vite + Material-UI
- **Key Files**:
  - `src/App.jsx`: Router setup
  - `src/components/HomePage.jsx`: Main layout
  - `src/components/ModelSection/ModelSection.jsx`: Analysis UI
  - `src/components/ModelSection/hooks/useImageAnalysis.js`: Logic hook
- **Dependencies**: Listed in `package.json`

#### Frontend/src/components/

- **Purpose**: Modular React components
- **Organization**: Feature-based grouping
- **Patterns**:
  - Component co-location (styles, logic, UI together)
  - Custom hooks for reusable logic
  - PropTypes for type checking
  - Material-UI theming

### Architecture Patterns

1. **Backend (Python)**:

   - **Singleton Pattern**: ModelManager class (single instance)
   - **Factory Pattern**: Model building based on string identifier
   - **Strategy Pattern**: Different preprocessing per model
   - **Dependency Injection**: Device, checkpoints configurable
   - **Error Handling**: Try-catch with logging, HTTP status codes

2. **Frontend (React)**:

   - **Component Composition**: Small, reusable components
   - **Custom Hooks**: useImageAnalysis for complex logic
   - **State Management**: React useState, useCallback, useEffect
   - **Prop Drilling Avoidance**: Hooks centralize state
   - **Responsive Design**: Mobile-first with breakpoints

3. **API Design**:
   - **RESTful**: Resource-based URLs (/models/, /predict/)
   - **Stateless**: Each request independent
   - **JSON**: Standard data format
   - **CORS**: Secure cross-origin access
   - **Versioning Ready**: URL structure allows /v1/, /v2/, etc.

## 🛠️ Technical Stack

### Backend Technologies

#### Core Framework

| Technology   | Version              | Purpose                         |
| ------------ | -------------------- | ------------------------------- |
| **Python**   | 3.8+                 | Primary programming language    |
| **FastAPI**  | 0.100+               | Modern async web framework      |
| **Uvicorn**  | 0.23+                | ASGI server for FastAPI         |
| **Pydantic** | (FastAPI dependency) | Data validation & serialization |

#### Deep Learning

| Technology      | Version | Purpose                         |
| --------------- | ------- | ------------------------------- |
| **PyTorch**     | 2.0+    | Deep learning framework         |
| **TorchVision** | 0.15+   | Pre-trained models & transforms |
| **timm**        | Latest  | PyTorch Image Models library    |
| **TorchCAM**    | 0.3+    | GradCAM implementation          |

#### Computer Vision & Image Processing

| Technology                 | Version | Purpose                       |
| -------------------------- | ------- | ----------------------------- |
| **OpenCV (opencv-python)** | 4.8+    | Image manipulation & heatmaps |
| **Pillow (PIL)**           | 9.0+    | Image loading & conversion    |
| **NumPy**                  | 1.21+   | Numerical operations          |

#### AI & LLM Integration

| Technology              | Version | Purpose           |
| ----------------------- | ------- | ----------------- |
| **google-generativeai** | 0.3+    | Gemini API client |

#### Data Science & Analysis

| Technology       | Version | Purpose                              |
| ---------------- | ------- | ------------------------------------ |
| **Pandas**       | 2.0+    | Data manipulation                    |
| **scikit-learn** | 1.0+    | ML utilities                         |
| **Matplotlib**   | 3.5+    | Visualization (optional)             |
| **Seaborn**      | 0.11+   | Statistical visualization (optional) |
| **tqdm**         | 4.64+   | Progress bars                        |

#### Utilities

| Technology           | Version | Purpose                         |
| -------------------- | ------- | ------------------------------- |
| **python-dotenv**    | 1.0+    | Environment variable management |
| **python-multipart** | 0.0.6+  | Multipart form data parsing     |

### Frontend Technologies

#### Core Framework

| Technology                   | Version | Purpose                     |
| ---------------------------- | ------- | --------------------------- |
| **React**                    | 19.1.1  | UI library                  |
| **React DOM**                | 19.1.1  | React renderer              |
| **Vite**                     | 7.1.6   | Build tool & dev server     |
| **@vitejs/plugin-react-swc** | 4.0.1   | Fast React refresh with SWC |

#### UI Library

| Technology                      | Version | Purpose           |
| ------------------------------- | ------- | ----------------- |
| **Material-UI (@mui/material)** | 7.3.2   | Component library |
| **@mui/icons-material**         | 7.3.2   | Icon library      |
| **@emotion/react**              | 11.14.0 | CSS-in-JS styling |
| **@emotion/styled**             | 11.14.1 | Styled components |

#### Routing

| Technology           | Version | Purpose             |
| -------------------- | ------- | ------------------- |
| **React Router DOM** | 7.9.1   | Client-side routing |

#### Code Quality & Linting

| Technology                      | Version | Purpose                     |
| ------------------------------- | ------- | --------------------------- |
| **ESLint**                      | 9.35.0  | JavaScript linting          |
| **@eslint/js**                  | 9.35.0  | ESLint core                 |
| **eslint-plugin-react-hooks**   | 5.2.0   | React hooks linting         |
| **eslint-plugin-react-refresh** | 0.4.20  | React refresh linting       |
| **globals**                     | 16.4.0  | Global variables definition |

#### TypeScript (Type Checking)

| Technology           | Version | Purpose                    |
| -------------------- | ------- | -------------------------- |
| **@types/react**     | 19.1.13 | React type definitions     |
| **@types/react-dom** | 19.1.9  | React DOM type definitions |

### Development Tools

#### Version Control

| Tool        | Purpose                       |
| ----------- | ----------------------------- |
| **Git**     | Source code version control   |
| **Git LFS** | Large file storage for models |
| **GitHub**  | Code hosting & collaboration  |

#### Environment Management

| Tool            | Purpose                    |
| --------------- | -------------------------- |
| **Python venv** | Python virtual environment |
| **Node.js**     | JavaScript runtime         |
| **npm**         | Node package manager       |

#### API Testing

| Tool                   | Purpose                       |
| ---------------------- | ----------------------------- |
| **FastAPI Swagger UI** | Interactive API documentation |
| **Redoc**              | Alternative API documentation |
| **curl**               | Command-line API testing      |

### External Services

#### APIs

| Service                     | Purpose                       |
| --------------------------- | ----------------------------- |
| **Google Gemini 2.5 Flash** | Natural language explanations |
| **Kaggle API**              | Dataset download              |

### Infrastructure (Recommended)

#### Compute

| Resource    | Recommended            | Purpose                   |
| ----------- | ---------------------- | ------------------------- |
| **CPU**     | 4+ cores               | Backend processing        |
| **RAM**     | 16 GB+                 | Model loading & inference |
| **GPU**     | NVIDIA CUDA-compatible | Accelerated inference     |
| **VRAM**    | 8 GB+                  | GPU model loading         |
| **Storage** | 20 GB+                 | Models + dataset          |

#### Deployment Options

| Platform              | Use Case                         |
| --------------------- | -------------------------------- |
| **Docker**            | Containerized deployment         |
| **AWS EC2**           | Cloud compute instance           |
| **Google Cloud Run**  | Serverless container deployment  |
| **Azure App Service** | Managed web hosting              |
| **Heroku**            | Simple cloud deployment          |
| **Vercel**            | Frontend hosting (Frontend only) |
| **Netlify**           | Frontend hosting (Frontend only) |

### Browser Compatibility

| Browser     | Minimum Version |
| ----------- | --------------- |
| **Chrome**  | 90+             |
| **Firefox** | 88+             |
| **Safari**  | 14+             |
| **Edge**    | 90+             |

### Operating System Compatibility

| OS          | Status                              |
| ----------- | ----------------------------------- |
| **Windows** | ✅ Tested (10, 11)                  |
| **macOS**   | ✅ Compatible (10.15+)              |
| **Linux**   | ✅ Compatible (Ubuntu 20.04+, etc.) |

---

**Continue to [Installation](#-installation) →**
