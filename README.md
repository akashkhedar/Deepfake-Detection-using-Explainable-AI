# 🧠 AI-Powered Deepfake Detection with Explainable AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-19.1.1-61DAFB.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.117+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A sophisticated full-stack deepfake detection system that combines state-of-the-art machine learning with explainable AI to identify AI-generated images. The system provides detailed visual explanations using GradCAM attention heatmaps and AI-powered contextual analysis.

## 🌟 Features

### 🎯 **Core Capabilities**

- **Real-time Deepfake Detection**: Binary classification of real vs. AI-generated images
- **Explainable AI**: GradCAM heatmaps showing which image regions influenced the model's decision
- **AI-Powered Explanations**: Dynamic explanations generated using Google Gemini AI that analyze attention regions
- **Interactive Web Interface**: Modern React frontend with drag-and-drop image upload
- **RESTful API**: FastAPI backend with comprehensive error handling

### 🔬 **Technical Features**

- **ResNet50 Architecture**: Pre-trained CNN fine-tuned for deepfake detection
- **GradCAM Visualization**: Visual attention maps highlighting suspicious regions
- **OpenCV Optimization**: High-performance image processing and heatmap generation
- **Mobile-Responsive Design**: Optimized UI for desktop and mobile devices
- **CORS-Enabled API**: Cross-origin support for frontend-backend communication

### 🤖 **AI Integration**

- **Google Gemini Integration**: Context-aware explanations based on model predictions
- **Multi-modal Analysis**: Combines visual heatmaps with textual explanations
- **Region-Aware Explanations**: AI analyzes which facial features triggered detection

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend │◄──►│   FastAPI Backend │◄──►│  ML Model + AI  │
│                 │    │                  │    │                 │
│ • Material-UI   │    │ • PyTorch Model  │    │ • ResNet50      │
│ • Drag & Drop   │    │ • GradCAM        │    │ • GradCAM       │
│ • Responsive    │    │ • OpenCV         │    │ • Gemini AI     │
│ • Modular       │    │ • CORS           │    │ • Explanations  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 📁 Project Structure

```
ML Project/
├── Backend/                    # FastAPI Server & ML Components
│   ├── main.py                # FastAPI application with Gemini integration
│   ├── train_resnet50.py       # Model training script (update: now uses ResNet50)
│   ├── test_resnet50.py        # Model evaluation script (update: now uses ResNet50)
│   ├── check_dataset.py        # Dataset validation utilities
│   ├── make_mini.py            # Mini dataset creator
│   ├── requirements.txt        # Python dependencies
│   ├── resnet18_best.pth       # Trained model weights (binary classifier -- filename kept for backward compatibility)
│   ├── .env                    # Environment variables (API keys)
│   └── venv/                   # Python virtual environment
│
├── Frontend/                   # React Application
│   ├── src/
│   │   ├── components/
│   │   │   ├── HomePage.jsx    # Main application container
│   │   │   ├── Navbar/         # Navigation component
│   │   │   └── ModelSection/   # Core detection interface
│   │   │       ├── ModelSection.jsx      # Main layout component
│   │   │       ├── ImageUploadArea.jsx   # Drag & drop upload
│   │   │       ├── ImageDisplay.jsx      # Image preview
│   │   │       ├── HeatmapDisplay.jsx    # GradCAM visualization
│   │   │       ├── ExplanationBox.jsx    # AI explanations
│   │   │       ├── WaitingState.jsx      # Loading states
│   │   │       └── hooks/
│   │   │           └── useImageAnalysis.js # Custom React hook
│   │   ├── App.jsx             # Application router
│   │   ├── main.jsx            # React entry point
│   │   └── index.css           # Global styles
│   ├── package.json            # Node.js dependencies
│   └── vite.config.js          # Vite build configuration
│
└── dataset/                    # Training Data (not included)
    ├── train/
    │   ├── Fake/
    │   └── Real/
    ├── validation/
    │   ├── Fake/
    │   └── Real/
    └── test/
        ├── Fake/
        └── Real/
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **CUDA-compatible GPU** (optional, for training)
- **Google Gemini API Key** ([Get here](https://makersuite.google.com/app/apikey))

### 1. Clone Repository

```bash
git clone https://github.com/akashkhedar/Deepfake-Detection-using-Explainable-AI.git
cd Deepfake-Detection-using-Explainable-AI
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd Backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your Gemini API key:
# GEMINI_API_KEY=your_actual_api_key_here

# Start the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory (new terminal)
cd Frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 4. Access Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🔧 Installation & Configuration

### Detailed Backend Setup

#### Environment Configuration

Create a `.env` file in the `Backend/` directory:

```env
# Gemini API Configuration
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Optional: OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Backend Configuration
BACKEND_HOST=localhost
BACKEND_PORT=8000

# Model Configuration
# NOTE: The codebase now uses ResNet50. The checkpoint filename in this repo remains
# `resnet18_best.pth` for backward compatibility; update MODEL_PATH if you rename the file.
MODEL_PATH=resnet18_best.pth
DEVICE=auto  # auto, cpu, cuda
```

#### Virtual Environment Setup

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Dataset Preparation

#### Using Your Own Dataset

1. **Organize your dataset**:

```
dataset/
├── train/
│   ├── Fake/      # AI-generated images
│   └── Real/      # Authentic images
├── validation/
│   ├── Fake/
│   └── Real/
└── test/
    ├── Fake/
    └── Real/
```

2. **Create a mini dataset for testing**:

```bash
python make_mini.py --src dataset --dst dataset_mini --per_class 2000
```

3. **Validate dataset integrity**:

```bash
python check_dataset.py --root dataset --make_sample_grid
```

## 🎯 Usage Guide

### Web Interface

1. **Open the application** at http://localhost:5173
2. **Upload an image** by:
   - Dragging and dropping into the upload area
   - Clicking "Choose Image" button
3. **View results**:
   - **Prediction**: Real or Fake classification
   - **Heatmap**: Visual attention map showing suspicious regions
   - **Explanation**: AI-generated contextual analysis

### API Usage

#### Predict Endpoint

```bash
# Using curl
curl -X POST "http://localhost:8000/predict/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.jpg"

# Response
{
  "prediction": "Fake",
  "heatmap": "data:image/png;base64,iVBORw0KGgoA...",
  "explanation": "The model detected inconsistencies in the facial region, particularly around the eyes and mouth, which are common artifacts in AI-generated images."
}
```

#### Python Client Example

```python
import requests

def predict_image(image_path):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:8000/predict/', files=files)
    return response.json()

result = predict_image('test_image.jpg')
print(f"Prediction: {result['prediction']}")
print(f"Explanation: {result['explanation']}")
```

## 🧠 Machine Learning Pipeline

### Model Architecture

- **Base Model**: ResNet18 pre-trained on ImageNet -**Base Model**: ResNet50 pre-trained on ImageNet
- **Modification**: Final layer adapted for binary classification (Real/Fake)
- **Input Size**: 224×224×3 RGB images
- **Output**: 2 classes with softmax probabilities

### Training Process

```bash
# Train the model
python train_resnet50.py

# Configuration options:
# - DATA_DIR: Dataset directory
# - BATCH_SIZE: Training batch size (default: 16)
# - NUM_EPOCHS: Training epochs (default: 8)
# - LR: Learning rate (default: 1e-4)
```

#### Training Features

- **Data Augmentation**: Random horizontal flip, color jitter
- **Mixed Precision**: AMP for faster training
- **AdamW Optimizer**: Weight decay regularization
- **Best Model Saving**: Automatic checkpoint saving

### Model Evaluation

```bash
# Evaluate the trained model
python test_resnet50.py
```

#### Evaluation Metrics

- **Classification Report**: Precision, Recall, F1-score
- **Confusion Matrix**: Visual performance analysis
- **Misclassification Analysis**: Error pattern identification
- **GradCAM Visualization**: Attention map analysis

### GradCAM Integration

The system uses GradCAM (Gradient-weighted Class Activation Mapping) to provide visual explanations:

```python
# GradCAM implementation
cam_extractor = GradCAM(model, target_layer=model.layer4[-1].conv3)
cam = cam_extractor(pred_class, output)[0]
```

## 🎨 Frontend Components

### Component Architecture

#### Core Components

1. **ModelSection.jsx**: Main container component
2. **ImageUploadArea.jsx**: Drag-and-drop file upload
3. **ImageDisplay.jsx**: Image preview with prediction status
4. **HeatmapDisplay.jsx**: GradCAM visualization
5. **ExplanationBox.jsx**: AI-generated explanations
6. **WaitingState.jsx**: Loading and empty states

#### Custom Hooks

**useImageAnalysis.js**: Centralized state management for:

- Image upload handling
- API communication
- Loading states
- Error handling

```javascript
const {
  selectedImage,
  prediction,
  heatmapUrl,
  explanation,
  isAnalyzing,
  handleImageUpload,
  handleFileSelect,
  handleReset,
} = useImageAnalysis();
```

### Responsive Design

- **Material-UI Integration**: Professional component library
- **Mobile-First Design**: Optimized for all device sizes
- **Dark Theme**: Modern dark color scheme
- **Accessibility**: ARIA labels and keyboard navigation

## 🔌 API Reference

### Endpoints

#### `GET /`

Health check endpoint

**Response:**

```json
{
  "message": "DeepFake Detection API is running"
}
```

#### `POST /predict/`

Analyze uploaded image for deepfake detection

**Parameters:**

- `file` (multipart/form-data): Image file (JPG, PNG, WebP)

**Response:**

```json
{
  "prediction": "Real|Fake",
  "heatmap": "data:image/png;base64,...",
  "explanation": "AI-generated explanation text"
}
```

**Error Response:**

```json
{
  "error": "Prediction failed: <error_message>"
}
```

### CORS Configuration

The API supports cross-origin requests from:

- localhost:3000 (React dev server)
- localhost:5173 (Vite dev server)
- localhost:4173 (Vite preview)
- 127.0.0.1 variants

## 🤖 AI Integration Details

### Google Gemini Integration

The system uses Google's Gemini AI model for generating contextual explanations:

```python
def call_gemini_with_heatmap(pred_class, overlay_base64):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([
        prompt,
        {"mime_type": "image/png", "data": base64.b64decode(overlay_base64)}
    ])
    return response.text.strip()
```

#### Features

- **Multi-modal Analysis**: Combines text prompts with heatmap images
- **Context-Aware**: Tailored explanations based on prediction type
- **Fallback Handling**: Graceful degradation when API is unavailable

### Explanation Generation

- **Fake Images**: Detailed analysis of suspicious regions with heatmap context
- **Real Images**: Simple confirmation with natural pattern identification
- **Error Cases**: Clear error messages with troubleshooting guidance

## 🔧 Development

### Frontend Development

```bash
# Start development server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

### Backend Development

```bash
# Start with auto-reload
uvicorn main:app --reload

# Run tests
python test_resnet18.py

# Create mini dataset
python make_mini.py --per_class 1000

# Validate dataset
python check_dataset.py --make_sample_grid
```

### Adding New Features

#### Frontend Component Creation

1. Create component in `src/components/`
2. Add PropTypes for type checking
3. Implement responsive design with Material-UI
4. Export from `index.js` for clean imports

#### Backend Endpoint Addition

1. Add route in `main.py`
2. Implement request/response models
3. Add error handling
4. Update CORS if needed

## 📊 Performance Optimization

### Backend Optimizations

- **OpenCV Integration**: 3x faster heatmap generation vs. matplotlib
- **Mixed Precision Training**: Reduced GPU memory usage
- **Efficient Image Processing**: Optimized tensor operations
- **Base64 Encoding**: Direct image transfer without file I/O

### Frontend Optimizations

- **Vite Build System**: Fast development and optimized production builds
- **React 19**: Latest performance improvements
- **Component Memoization**: Reduced re-renders
- **Lazy Loading**: Code splitting for faster initial load

## 🐛 Troubleshooting

### Common Issues

#### Backend Issues

**1. Model Not Found**

```
FileNotFoundError: [Errno 2] No such file or directory: 'resnet18_best.pth'
```

**Solution**: Ensure the trained model file exists or retrain the model.

**2. CUDA Issues**

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size or use CPU mode by setting `DEVICE = "cpu"`.

**3. Gemini API Issues**

```
google.api_core.exceptions.PermissionDenied: 403 API key not valid
```

**Solution**: Verify your Gemini API key in the `.env` file.

#### Frontend Issues

**1. CORS Errors**

```
Access to fetch at 'http://localhost:8000' from origin 'http://localhost:5173' has been blocked by CORS policy
```

**Solution**: Ensure backend CORS middleware includes your frontend URL.

**2. API Connection Failed**

```
Failed to analyze the image. Please make sure the backend server is running
```

**Solution**: Verify backend is running on http://localhost:8000.

### Environment Setup Issues

**1. Virtual Environment**

```bash
# Windows: If activation fails
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# macOS/Linux: If permission denied
chmod +x venv/bin/activate
```

**2. Package Installation**

```bash
# If pip install fails
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

## 🔐 Security Considerations

### API Security

- **File Upload Validation**: Only image files accepted
- **File Size Limits**: Prevent large file uploads
- **Input Sanitization**: Secure file handling
- **Error Message Sanitization**: Prevent information leakage

### Environment Variables

- **API Key Protection**: Never commit `.env` files
- **Access Control**: Restrict API access in production
- **HTTPS**: Use secure connections in production

## 📈 Future Enhancements

### Planned Features

- **🎥 Video Analysis**: Frame-by-frame deepfake detection
- **📱 Mobile App**: Native iOS/Android applications
- **🔍 Batch Processing**: Multiple image analysis
- **📊 Advanced Analytics**: Detailed metrics dashboard
- **🌐 Multi-language Support**: Internationalization
- **🏢 Enterprise Features**: User management and API rate limiting

### Model Improvements

- **🧠 Advanced Architectures**: EfficientNet, Vision Transformers
- **📚 Larger Datasets**: Training on more diverse data
- **🎯 Multi-class Detection**: Specific deepfake technique identification
- **⚡ Real-time Processing**: Optimized inference pipeline

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes** and **add tests**
4. **Commit changes**: `git commit -m 'Add amazing feature'`
5. **Push to branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Code Standards

- **Python**: Follow PEP 8 style guide
- **JavaScript**: Use ESLint configuration
- **Documentation**: Update README for new features
- **Testing**: Add tests for new functionality

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team**: For the deep learning framework
- **Hugging Face**: For model architectures and inspiration
- **Material-UI Team**: For the React component library
- **FastAPI Developers**: For the modern Python web framework
- **Google**: For the Gemini AI API
- **Research Community**: For GradCAM and explainable AI techniques

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/akashkhedar/Deepfake-Detection-using-Explainable-AI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/akashkhedar/Deepfake-Detection-using-Explainable-AI/discussions)
- **Email**: akashkhedar262@gmail.com

## 📊 Project Stats

- **🔧 Languages**: Python, JavaScript, CSS, HTML
- **📚 Framework**: PyTorch, React, FastAPI
- **🎯 Best Validation Accuracy**: 98.57% (after 8 epochs)
- **🧪 Test Accuracy**: 88% on the held-out test set
- **⚡ Performance**: <2s average inference time
- **📱 Compatibility**: Desktop, tablet, mobile responsive

## 🧾 Training Results

Below are the training and validation metrics recorded during the 8-epoch training run (logs captured from the training terminal). The model used: ResNet50 (final layer adapted for binary classification).

Epoch-by-epoch summary:

- Epoch 1/8 — Train Loss: 0.0691 Acc: 0.9737 | Val Loss: 0.0738 Acc: 0.9727 (Checkpoint saved)
- Epoch 2/8 — Train Loss: 0.0417 Acc: 0.9837 | Val Loss: 0.0956 Acc: 0.9620 (Checkpoint saved)
- Epoch 3/8 — Train Loss: 0.0341 Acc: 0.9862 | Val Loss: 0.0644 Acc: 0.9772 (Best so far)
- Epoch 4/8 — Train Loss: 0.0307 Acc: 0.9876 | Val Loss: 0.0475 Acc: 0.9827 (Best so far)
- Epoch 5/8 — Train Loss: 0.0269 Acc: 0.9891 | Val Loss: 0.0653 Acc: 0.9759
- Epoch 6/8 — Train Loss: 0.0245 Acc: 0.9900 | Val Loss: 0.0544 Acc: 0.9797
- Epoch 7/8 — Train Loss: 0.0220 Acc: 0.9910 | Val Loss: 0.0478 Acc: 0.9844 (Best so far)
- Epoch 8/8 — Train Loss: 0.0206 Acc: 0.9918 | Val Loss: 0.0438 Acc: 0.9857 (Final best)

- Training complete. Best Validation Accuracy: 0.9857 (98.57%)

Final classification report on the test set (10905 samples):

        precision    recall  f1-score   support

    Fake       0.81      0.98      0.89      5492
    Real       0.97      0.77      0.86      5413

accuracy 0.88 10905
macro avg 0.89 0.88 0.88 10905
weighted avg 0.89 0.88 0.88 10905

Interpretation and notes:

- The model achieves very high validation accuracy (98.57%) during training, indicating strong performance on the held-out validation split.
- On the test set, overall accuracy is 88%. The class-wise breakdown shows the model is highly precise at detecting Real images (precision 0.97) but has lower recall for Real (0.77). Conversely, Fake images have high recall (0.98) but lower precision (0.81). This indicates the model tends to prefer predicting "Fake" slightly more often, catching most fakery but producing more false positives for the Fake class.
- The macro- and weighted-average F1-scores (~0.88–0.89) indicate balanced performance across classes when accounting for both precision and recall.
- Actionable next steps: tune class weights or thresholding to improve Real-class recall, collect more balanced/ diverse Real examples, and experiment with additional architectures or augmentation strategies.

---

**Built with ❤️ by [Akash Khedar](https://github.com/akashkhedar)**

_Making AI explainable, one detection at a time._
