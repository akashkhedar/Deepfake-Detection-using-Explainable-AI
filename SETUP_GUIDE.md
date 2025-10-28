# Complete Setup Guide

This guide will walk you through setting up the Deepfake Detection project from scratch.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Step 1: Clone the Repository](#step-1-clone-the-repository)
- [Step 2: Download Model Weights](#step-2-download-model-weights)
- [Step 3: Download the Dataset](#step-3-download-the-dataset)
- [Step 4: Backend Setup](#step-4-backend-setup)
- [Step 5: Frontend Setup](#step-5-frontend-setup)
- [Step 6: Start the Application](#step-6-start-the-application)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before you begin, ensure you have the following installed:

### Required Software
- **Python 3.8+**: [Download](https://www.python.org/downloads/)
- **Node.js 16+**: [Download](https://nodejs.org/)
- **Git**: [Download](https://git-scm.com/downloads)
- **Git LFS**: [Download](https://git-lfs.com/) - **IMPORTANT for model files**

### Optional but Recommended
- **CUDA Toolkit** (for GPU acceleration): [Download](https://developer.nvidia.com/cuda-downloads)
- **Kaggle Account** (for dataset download): [Sign up](https://www.kaggle.com/)
- **Google Gemini API Key** (for AI explanations): [Get API Key](https://makersuite.google.com/app/apikey)

### Verify Installations

```bash
# Check Python
python --version  # Should be 3.8 or higher

# Check Node.js
node --version    # Should be 16 or higher

# Check Git
git --version

# Check Git LFS
git lfs version   # If not installed, install from https://git-lfs.com/
```

## Step 1: Clone the Repository

### 1.1 Install Git LFS (If Not Already Installed)

**Windows:**
```powershell
# Download and run installer from https://git-lfs.com/
# Or using Chocolatey:
choco install git-lfs

# Or using Scoop:
scoop install git-lfs
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# Fedora
sudo dnf install git-lfs

# Arch Linux
sudo pacman -S git-lfs
```

**macOS:**
```bash
# Using Homebrew
brew install git-lfs
```

### 1.2 Initialize Git LFS

```bash
git lfs install
```

### 1.3 Clone the Repository

```bash
# Clone with HTTPS
git clone https://github.com/yourusername/Deepfake-Detection-using-Explainable-AI.git

# Or with SSH
git clone git@github.com:yourusername/Deepfake-Detection-using-Explainable-AI.git

# Navigate to the project
cd Deepfake-Detection-using-Explainable-AI
```

## Step 2: Download Model Weights

If you cloned with Git LFS properly installed, model files should download automatically. Verify:

### 2.1 Verify Model Files

```bash
# Check that model files are actual binaries, not text pointers
cd Backend/models

# Linux/Mac
ls -lh *.pth
# Each file should be 75-240 MB

# Windows PowerShell
Get-ChildItem *.pth | Format-Table Name, @{L="Size (MB)";E={[math]::Round($_.Length/1MB, 2)}}
```

### 2.2 If Files Are Pointer Files (< 1 MB)

If the files are small (< 1 MB), they're pointer files. Download the actual files:

```bash
# From the project root
git lfs pull
```

### 2.3 Manual Download (If Git LFS Fails)

If Git LFS doesn't work, you can download models from alternative sources:

```bash
# Coming soon - alternative download links will be provided
# For now, ensure Git LFS is working properly
```

## Step 3: Download the Dataset

The dataset is **not included** in the repository due to size (~2.3 GB).

### 3.1 Install Kaggle CLI

```bash
pip install kaggle
```

### 3.2 Configure Kaggle Credentials

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json`

5. Place the file:
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

6. Set permissions (Linux/Mac only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 3.3 Download the Dataset

```bash
cd Backend

# Download dataset
kaggle datasets download -d xhlulu/140k-real-and-fake-faces

# Extract
# Linux/Mac
unzip 140k-real-and-fake-faces.zip -d dataset/

# Windows PowerShell
Expand-Archive -Path 140k-real-and-fake-faces.zip -DestinationPath dataset/

# Clean up zip file (optional)
rm 140k-real-and-fake-faces.zip
```

### 3.4 Verify Dataset Structure

```bash
# Should show: Train/, Validation/, Test/
ls dataset/

# Check counts
ls dataset/Train/Fake/ | wc -l     # Should be ~70,001
ls dataset/Train/Real/ | wc -l     # Should be ~70,001
```

## Step 4: Backend Setup

### 4.1 Navigate to Backend

```bash
cd Backend  # If not already there
```

### 4.2 Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate it
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

Your prompt should now show `(venv)` prefix.

### 4.3 Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- PyTorch and TorchVision
- FastAPI and Uvicorn
- OpenCV and Pillow
- TorchCAM
- Google Generative AI
- And more...

**Note**: Installation may take several minutes, especially for PyTorch.

### 4.4 Configure Environment Variables

Create a `.env` file in the `Backend` directory:

```bash
# Create .env file
# Linux/Mac
cat > .env << EOF
GEMINI_API_KEY=your_gemini_api_key_here
DEVICE=cuda
LOAD_MODELS=all
EOF

# Windows (PowerShell)
@"
GEMINI_API_KEY=your_gemini_api_key_here
DEVICE=cuda
LOAD_MODELS=all
"@ | Out-File -FilePath .env -Encoding utf8
```

**Important**: Replace `your_gemini_api_key_here` with your actual API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

If you don't have a GPU, change `DEVICE=cuda` to `DEVICE=cpu`.

### 4.5 Verify Backend Setup

```bash
# Quick test - should show no errors
python -c "import torch; import fastapi; import cv2; print('All imports successful!')"

# Check GPU availability (if you have one)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 5: Frontend Setup

### 5.1 Navigate to Frontend

```bash
# From Backend directory
cd ../Frontend

# Or from project root
cd Frontend
```

### 5.2 Install Node Dependencies

```bash
# Using npm
npm install

# Or using yarn
yarn install
```

This will install:
- React and React Router
- Material-UI
- Vite
- And more...

### 5.3 Configure Frontend Environment (Optional)

If your backend runs on a different port or host:

```bash
# Create .env file
# Linux/Mac
echo "VITE_API_BASE=http://localhost:8000" > .env

# Windows (PowerShell)
"VITE_API_BASE=http://localhost:8000" | Out-File -FilePath .env -Encoding utf8
```

## Step 6: Start the Application

### 6.1 Start the Backend

Open a terminal and:

```bash
cd Backend
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Loading model: resnet50
INFO:     Loading model: resnet152
...
INFO:     Application startup complete.
```

**Note**: First startup may take 1-2 minutes to load all models into memory.

### 6.2 Start the Frontend

Open a **new terminal** and:

```bash
cd Frontend
npm run dev  # or yarn dev
```

You should see:
```
  VITE v7.1.6  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### 6.3 Open the Application

Open your browser and navigate to:
```
http://localhost:5173
```

You should see the Deepfake Detection interface!

### 6.4 Test the Application

1. Click "Choose Image" or drag and drop an image
2. Wait for analysis (takes 5-15 seconds)
3. View results:
   - Prediction (Real/Fake)
   - Confidence score
   - Heatmap overlay
   - AI explanation

## Troubleshooting

### Issue: Git LFS files are pointers, not actual files

**Symptoms**: Model files are < 1 KB in size

**Solution**:
```bash
git lfs install
git lfs pull
```

### Issue: Kaggle authentication failed

**Symptoms**: `401 Unauthorized` when downloading dataset

**Solution**:
- Verify `kaggle.json` is in the correct location
- Check file permissions: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)
- Ensure you're logged into Kaggle.com and have accepted competition rules

### Issue: CUDA out of memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Load fewer models
export LOAD_MODELS=resnet50,xception  # Linux/Mac
$env:LOAD_MODELS="resnet50,xception"  # Windows PowerShell

# Or force CPU mode
export DEVICE=cpu  # Linux/Mac
$env:DEVICE="cpu"  # Windows PowerShell
```

### Issue: Module not found errors

**Symptoms**: `ModuleNotFoundError: No module named 'xxx'`

**Solution**:
```bash
# Ensure virtual environment is activated
# Then reinstall dependencies
pip install -r requirements.txt
```

### Issue: Port already in use

**Symptoms**: `Address already in use` or `port 8000 already in use`

**Solution**:
```bash
# Use a different port
uvicorn main:app --reload --port 8001

# Update frontend .env
echo "VITE_API_BASE=http://localhost:8001" > Frontend/.env
```

### Issue: Frontend shows "Failed to fetch"

**Symptoms**: Network errors in browser console

**Solution**:
1. Verify backend is running: `curl http://localhost:8000/`
2. Check CORS settings in `Backend/main.py`
3. Verify frontend `.env` has correct API URL

### Issue: Models load slowly

**Symptoms**: Backend takes 2+ minutes to start

**Solution**: This is normal for the first run. Subsequent runs use cached models and are faster.

### Issue: Gemini API errors

**Symptoms**: Explanations show generic text instead of AI-generated

**Solution**:
- Verify `GEMINI_API_KEY` in `.env`
- Check API quota at [Google AI Studio](https://makersuite.google.com/)
- Ensure you have an active API key

## Next Steps

Once everything is working:

1. **Explore the UI**: Try different images, models, and settings
2. **Check the API**: Visit `http://localhost:8000/docs` for API documentation
3. **Review the code**: Understand how the ensemble works
4. **Customize**: Modify models, add features, improve UI
5. **Deploy**: Consider Docker, cloud deployment, etc.

## Getting Help

- **Issues**: Open an issue on GitHub
- **Documentation**: Check `README.md` and other docs
- **Community**: Join discussions on GitHub

---

Congratulations! You now have a fully functional deepfake detection system. 🎉
