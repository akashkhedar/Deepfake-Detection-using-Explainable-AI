# backend/main.py

# Standard library imports
import io
import base64
import os

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

# Env + Gemini
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes: Real / Fake

state_dict = torch.load("resnet18_best.pth", map_location=DEVICE)
model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()

# Enable grads for GradCAM
for param in model.parameters():
    param.requires_grad_(True)

cam_extractor = GradCAM(model, target_layer=model.layer4[-1].conv2)

# Preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def get_cam_overlay(img_tensor, cam, original_size):
    if cam.dim() == 3:
        cam = cam.squeeze(0)

    cam = cam.detach().cpu().numpy().astype(np.float32)
    # Resize CAM to match original image size instead of 224x224
    cam = cv2.resize(cam, original_size, interpolation=cv2.INTER_LINEAR)

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Use original image data instead of processed tensor
    img_normalized = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_denormalized = np.clip(std * img_normalized + mean, 0, 1)

    # Resize denormalized image to original size
    img_original_size = cv2.resize(
        img_denormalized, original_size, interpolation=cv2.INTER_LINEAR
    )

    overlay = 0.4 * heatmap + 0.6 * img_original_size
    overlay = np.clip(overlay, 0, 1)

    overlay_uint8 = (overlay * 255).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(
        ".png", overlay_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6]
    )

    if not success:
        raise Exception("Failed to encode image")

    return base64.b64encode(buffer).decode("utf-8")


def call_gemini_with_heatmap(pred_class, overlay_base64):
    prediction_text = "Fake" if pred_class == 0 else "Real"

    prompt = f"""
    The AI model analyzed an image and predicted it is {prediction_text}.
    Here is the heatmap showing which areas of the image the model focused on.
    Please explain in simple words (2-3 sentences) why the image is likely {prediction_text.lower()}.
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            [
                prompt,
                {"mime_type": "image/png", "data": base64.b64decode(overlay_base64)},
            ]
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini call failed: {e}")
        return f"Prediction: {prediction_text} (could not generate explanation)"


@app.get("/")
async def root():
    return {"message": "DeepFake Detection API is running"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        original_size = image.size  # (width, height)
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        input_tensor.requires_grad_(True)

        # Forward pass
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()

        # GradCAM
        cam = cam_extractor(pred_class, output)[0]
        overlay_base64 = get_cam_overlay(input_tensor.cpu()[0], cam, original_size)

        # Explanation
        if pred_class == 0:  # Fake → use Gemini with heatmap
            explanation = call_gemini_with_heatmap(pred_class, overlay_base64)
        else:  # Real → simple static explanation
            explanation = "The image appears natural with consistent details, so it is likely real."

        response_data = {
            "prediction": "Real" if pred_class == 1 else "Fake",
            "heatmap": overlay_base64,
            "explanation": explanation,
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
