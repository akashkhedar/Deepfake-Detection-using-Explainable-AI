# Dataset Information

## Overview

This project uses a deepfake detection dataset containing real and AI-generated images.

## Dataset Source

**Kaggle Dataset**: [140k Real and Fake Faces](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

## Download Instructions

### Option 1: Using Kaggle CLI (Recommended)

1. **Install Kaggle CLI**:

   ```bash
   pip install kaggle
   ```

2. **Set up Kaggle API credentials**:

   - Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
   - Click "Create New API Token" to download `kaggle.json`
   - Place it in:
     - **Linux/Mac**: `~/.kaggle/kaggle.json`
     - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

3. **Download the dataset**:

   ```bash
   cd Backend
   kaggle datasets download -d xhlulu/140k-real-and-fake-faces
   ```

4. **Extract the dataset**:

   ```bash
   # Linux/Mac
   unzip 140k-real-and-fake-faces.zip -d dataset/

   # Windows (PowerShell)
   Expand-Archive -Path 140k-real-and-fake-faces.zip -DestinationPath dataset/
   ```

### Option 2: Manual Download

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
2. Click the "Download" button (requires Kaggle account)
3. Extract the downloaded ZIP file to the `Backend/dataset/` directory

## Expected Dataset Structure

After extraction, your `dataset` directory should look like this:

```
Backend/dataset/
├── Train/
│   ├── Fake/          # 70,001 fake images
│   └── Real/          # 70,001 real images
├── Validation/
│   ├── Fake/          # 19,641 fake images
│   └── Real/          # 19,787 real images
└── Test/
    ├── Fake/          # 5,492 fake images
    └── Real/          # 5,413 real images
```

## Dataset Statistics

| Split      | Fake Images | Real Images | Total       |
| ---------- | ----------- | ----------- | ----------- |
| Train      | 70,001      | 70,001      | 140,002     |
| Validation | 19,641      | 19,787      | 39,428      |
| Test       | 5,492       | 5,413       | 10,905      |
| **Total**  | **95,134**  | **95,201**  | **190,335** |

## Dataset Properties

- **Image Format**: JPEG
- **Image Resolution**: 256×256 pixels
- **Color Space**: RGB
- **File Naming**:
  - Fake images: `fake_*.jpg`
  - Real images: `real_*.jpg`

## Verification

After downloading, you can verify the dataset integrity using the provided utility:

```bash
cd Backend
python check_dataset.py --root dataset --out dataset_summary.json --make_sample_grid
```

This will:

- Count images in each split and class
- Check for corrupted images
- Generate sample thumbnail grids
- Create a summary JSON file

## License & Citation

Please refer to the [Kaggle dataset page](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) for:

- Dataset license information
- Citation requirements
- Original source attribution

## Notes

- The dataset is **not included** in this repository due to size (~2.3 GB)
- You must download it separately following the instructions above
- Ensure you have sufficient disk space (~3 GB free recommended)
- The download and extraction may take several minutes depending on your internet connection
