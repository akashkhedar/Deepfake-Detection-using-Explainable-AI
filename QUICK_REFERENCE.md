# Quick Reference

Quick commands and tips for working with the Deepfake Detection project.

## 🚀 Quick Start

```bash
# 1. Clone with Git LFS
git lfs install
git clone <repo-url>
cd "ML Project"

# 2. Download dataset (requires Kaggle setup)
cd Backend
kaggle datasets download -d xhlulu/140k-real-and-fake-faces
unzip 140k-real-and-fake-faces.zip -d dataset/

# 3. Setup Backend
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# 4. Setup Frontend
cd ../Frontend
npm install

# 5. Start Backend (terminal 1)
cd Backend
source venv/bin/activate  # Windows: .\venv\Scripts\activate
uvicorn main:app --reload --port 8000

# 6. Start Frontend (terminal 2)
cd Frontend
npm run dev
```

## 📦 Git LFS Commands

```bash
# Install Git LFS
git lfs install

# Download all LFS files
git lfs pull

# List LFS files
git lfs ls-files

# Check LFS file status
git lfs status

# Track new file types
git lfs track "*.pth"

# Migrate existing files to LFS
git lfs migrate import --include="*.pth"
```

## 🔧 Backend Commands

```bash
# Activate virtual environment
source venv/bin/activate          # Linux/Mac
.\venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Start development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Start production server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Check API health
curl http://localhost:8000/

# View API docs
# Open browser: http://localhost:8000/docs

# Validate dataset
python check_dataset.py --root dataset --out dataset_summary.json --make_sample_grid

# Test imports
python -c "import torch; import fastapi; import cv2; print('OK')"

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🎨 Frontend Commands

```bash
# Install dependencies
npm install
# or
yarn install

# Start development server
npm run dev
# or
yarn dev

# Build for production
npm run build
# or
yarn build

# Preview production build
npm run preview
# or
yarn preview

# Lint code
npm run lint
# or
yarn lint
```

## 🔐 Environment Variables

### Backend `.env`

```env
GEMINI_API_KEY=your_api_key_here
DEVICE=cuda                    # or 'cpu'
LOAD_MODELS=all                # or 'resnet50,xception'
```

### Frontend `.env`

```env
VITE_API_BASE=http://localhost:8000
```

## 🧪 Testing & Debugging

```bash
# Test single endpoint
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"

# Test with specific model
curl -X POST "http://localhost:8000/predict/?model=resnet50" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"

# Get available models
curl http://localhost:8000/models/

# Get server status
curl http://localhost:8000/status/

# View logs
# Backend logs appear in terminal where uvicorn is running

# Frontend logs
# Open browser console (F12) when using the app
```

## 📊 Dataset Commands

```bash
# Download from Kaggle
kaggle datasets download -d xhlulu/140k-real-and-fake-faces

# Extract
unzip 140k-real-and-fake-faces.zip -d dataset/  # Linux/Mac
Expand-Archive 140k-real-and-fake-faces.zip -DestinationPath dataset/  # Windows

# Validate dataset
python check_dataset.py --root dataset --sample_per_class 12 --out dataset_summary.json

# Generate sample grids
python check_dataset.py --root dataset --make_sample_grid

# Count images
find dataset/Train/Fake -type f | wc -l    # Linux/Mac
(Get-ChildItem dataset/Train/Fake -File).Count  # Windows PowerShell
```

## 🐛 Common Issues & Fixes

### Git LFS pointer files instead of actual files

```bash
git lfs install
git lfs pull
```

### Port already in use

```bash
# Use different port
uvicorn main:app --reload --port 8001

# Or kill process using port (Linux/Mac)
lsof -ti:8000 | xargs kill -9

# Or kill process (Windows)
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process
```

### CUDA out of memory

```bash
# Load fewer models
export LOAD_MODELS=resnet50,xception  # Linux/Mac
$env:LOAD_MODELS="resnet50,xception"  # Windows PowerShell

# Or use CPU
export DEVICE=cpu  # Linux/Mac
$env:DEVICE="cpu"  # Windows PowerShell
```

### Module not found

```bash
# Ensure venv is activated, then:
pip install -r requirements.txt
```

### Node modules issues

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json  # Linux/Mac
Remove-Item -Recurse -Force node_modules, package-lock.json  # Windows PowerShell

npm install
```

## 🔍 Useful Checks

```bash
# Verify Python version
python --version  # Should be 3.8+

# Verify Node version
node --version    # Should be 16+

# Verify Git LFS
git lfs version

# Check virtual environment
which python      # Should point to venv  (Linux/Mac)
where python      # Should point to venv  (Windows)

# Check installed Python packages
pip list

# Check Node packages
npm list --depth=0

# Check model file sizes
ls -lh Backend/models/*.pth  # Linux/Mac
Get-ChildItem Backend/models/*.pth | Format-Table Name, @{L="Size (MB)";E={[math]::Round($_.Length/1MB, 2)}}  # Windows

# Check dataset structure
tree Backend/dataset -L 2  # Linux/Mac (if tree installed)
Get-ChildItem Backend/dataset -Recurse -Depth 2  # Windows PowerShell
```

## 📝 Git Workflow

```bash
# Check status
git status

# Stage changes
git add .

# Commit (LFS files handled automatically)
git commit -m "Description of changes"

# Push (LFS files uploaded automatically)
git push

# Pull with LFS files
git pull
git lfs pull  # If needed

# Create branch
git checkout -b feature/new-feature

# Switch branches
git checkout main

# View LFS file info
git lfs ls-files
```

## 🚢 Production Deployment

```bash
# Backend production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Frontend production build
cd Frontend
npm run build
# Serve the dist/ folder with your web server

# Using environment-specific configs
export GEMINI_API_KEY=prod_key  # Linux/Mac
$env:GEMINI_API_KEY="prod_key"  # Windows PowerShell
```

## 📚 Documentation Links

- [Main README](README.md)
- [Setup Guide](SETUP_GUIDE.md)
- [Dataset Info](Backend/DATASET.md)
- [Model Info](Backend/models/README.md)
- [FastAPI Docs](http://localhost:8000/docs) (when server is running)

## 🆘 Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/repo/issues)
- **API Docs**: http://localhost:8000/docs
- **React DevTools**: Install browser extension for debugging

---

**Tip**: Bookmark this file for quick access to common commands! 📌
