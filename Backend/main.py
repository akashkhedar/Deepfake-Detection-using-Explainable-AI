# backend/main.py

# Standard library imports
import io
import base64

# Third-party imports
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms, models
from torchcam.methods import GradCAM

# FastAPI imports
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:4173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:4173",
    ],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model architecture
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes: Real / Fake

# Load weights
state_dict = torch.load("resnet18_best.pth", map_location=DEVICE)
model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()

# Enable gradients for GradCAM
for param in model.parameters():
    param.requires_grad_(True)

# Use the last convolutional layer for GradCAM
cam_extractor = GradCAM(model, target_layer=model.layer4[-1].conv2)

# Preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def get_cam_overlay(img_tensor, cam):
    # cam: [1, H, W] or [H, W]
    if cam.dim() == 3:  # [1, H, W]
        cam = cam.squeeze(0)

    # Convert to numpy and resize using OpenCV (fastest)
    cam = cam.detach().cpu().numpy().astype(np.float32)  # <-- FIXED
    cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Normalize cam between 0 and 1
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Apply colormap using OpenCV (fastest)
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Denormalize image efficiently
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # <-- FIXED
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = np.clip(std * img + mean, 0, 1)

    # Fast blending
    overlay = 0.4 * heatmap + 0.6 * img
    overlay = np.clip(overlay, 0, 1)

    # Convert to PNG bytes using OpenCV
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(
        ".png", overlay_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6]
    )

    if not success:
        raise Exception("Failed to encode image")

    return base64.b64encode(buffer).decode("utf-8")


@app.get("/")
async def root():
    return {"message": "DeepFake Detection API is running"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        input_tensor.requires_grad_(True)  # Enable gradients for GradCAM

        # Forward pass for prediction
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()

        # Generate GradCAM
        cam = cam_extractor(pred_class, output)[0]
        overlay_base64 = get_cam_overlay(input_tensor.cpu()[0], cam)

        response_data = {
            "prediction": "Real" if pred_class == 1 else "Fake",
            "heatmap": overlay_base64,
        }

        return JSONResponse(
            content=response_data,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            },
        )

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return JSONResponse(
            content={"error": f"Prediction failed: {str(e)}"},
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            },
        )
