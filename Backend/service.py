import os
import base64
import logging
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torchcam.methods import GradCAM
import timm

from dotenv import load_dotenv
import google.generativeai as genai

# =========================================================
# ✅ Global Config
# =========================================================
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
load_dotenv()
logger = logging.getLogger("service")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = {
    "resnet50": "./models/resnet50.pth",
    "resnet152": "./models/resnet152v2.pth",
    "inceptionresnet": "./models/inceptionresnetv2.pth",
    "xceptionnet": "./models/xception.pth",
    "efficientnetb4": "./models/efficientnetb4.pth",
}

IMG_SIZE_MAP = {
    "resnet50": 224,
    "resnet152": 224,
    "inceptionresnet": 299,
    "xceptionnet": 299,
    "efficientnetb4": 380,
}


def make_preprocess(size=224):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# =========================================================
# ✅ Model Manager Class
# =========================================================
# =========================================================
# ✅ Model Manager Class
# =========================================================
class ModelManager:
    def __init__(self, device: str = DEVICE, checkpoints: dict = CHECKPOINTS):
        self.device = device
        self.checkpoints = checkpoints
        self.models_store: Dict[str, torch.nn.Module] = {}
        self.cam_extractors: Dict[str, GradCAM] = {}
        self.preprocess_store: Dict[str, transforms.Compose] = {}
        self.model_status: Dict[str, dict] = {}
        self.has_gemini = False # New flag for Gemini status

        # Optional Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                logger.info("✅ Gemini API configured.")
                self.has_gemini = True
            except Exception as e:
                logger.warning(f"⚠️ Failed to configure Gemini: {e}")

    # =====================================================
    # ✅ Model Factory (NO CHANGE)
    # =====================================================
    # ... (self.build_model remains exactly the same as your script) ...

    def build_model(self, name: str, num_classes: int = 2):
        name = name.lower()

        if name == "resnet50":
            m = models.resnet50(weights=None)
            m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
            target_layer = m.layer4[-1].conv2

        elif name == "resnet152":
            m = models.resnet152(weights=None)
            m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
            target_layer = m.layer4[-1].conv2

        elif name == "inceptionresnet":
            m = timm.create_model(
                "inception_resnet_v2", pretrained=False, num_classes=num_classes
            )
            target_layer = m.conv2d_7b  # last conv block

        elif name == "xceptionnet":
            import torch.nn as nn

            m = timm.create_model("legacy_xception", pretrained=False, num_classes=0)
            num_ftrs = m.num_features
            m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 1), nn.Sigmoid())

            old_forward = m.forward

            def wrapped_forward(x):
                out = old_forward(x)
                if out.ndim == 2 and out.shape[1] == 1:
                    out = torch.cat([1 - out, out], dim=1)
                return out

            m.forward = wrapped_forward
            target_layer = m.conv4  # fixed SeparableConv2d bug

        elif name == "efficientnetb4":
            m = timm.create_model(
                "efficientnet_b4", pretrained=False, num_classes=num_classes
            )
            target_layer = m.blocks[-1][-1].conv_pwl

        else:
            raise ValueError(f"Unknown model name: {name}")

        return m, target_layer

    # =====================================================
    # ✅ Checkpoint Loader (NO CHANGE)
    # =====================================================
    # ... (self.load_checkpoint remains exactly the same as your script) ...

    def load_checkpoint(self, model: torch.nn.Module, path: str, device: str = None):
        device = device or self.device
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except Exception:
            ckpt = torch.load(path, map_location=device)

        if ckpt is None:
            raise ValueError(f"Empty checkpoint: {path}")

        if isinstance(ckpt, dict):
            state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        else:
            state = ckpt

        if not hasattr(state, "items"):
            raise ValueError(f"Invalid state dict in checkpoint: {path}")

        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)
        return model

    # =====================================================
    # ✅ Load All Models (NO CHANGE)
    # =====================================================
    # ... (self.load_models remains exactly the same as your script) ...

    def load_models(
        self, names: Optional[List[str]] = None, to_device: Optional[str] = None
    ):
        to_device = to_device or self.device
        names = names or list(self.checkpoints.keys())

        for name in names:
            status = {
                "built": False,
                "checkpoint_loaded": False,
                "available": False,
                "errors": [],
            }
            try:
                m, target_layer = self.build_model(name)
                status["built"] = True
                m = self.load_checkpoint(m, self.checkpoints[name], device=to_device)
                status["checkpoint_loaded"] = True
                m.to(to_device).eval()

                cam = GradCAM(m, target_layer=target_layer)
                self.models_store[name] = m
                self.cam_extractors[name] = cam
                self.preprocess_store[name] = make_preprocess(IMG_SIZE_MAP[name])
                status["available"] = True
                logger.info(f"✅ Loaded model: {name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {name}: {e}")
                status["errors"].append(str(e))

            self.model_status[name] = status

    # =====================================================
    # ✅ Model Access Helpers (NO CHANGE)
    # =====================================================
    # ... (self.get_loaded_models and self.get_status remain the same) ...

    def get_loaded_models(self):
        return list(self.models_store.keys())

    def get_status(self):
        return {
            "device": self.device,
            "models": self.model_status,
            "available_models": self.get_loaded_models(),
        }

    # =====================================================
    # ✅ Utility Helpers (Refactored/New)
    # =====================================================

    def _get_cam_overlay_and_normalize(self, image_tensor: torch.Tensor, cam_tensor: torch.Tensor, target_size: tuple):
        """
        Creates an image overlay (heatmap + original image) and normalizes the CAM.
        FIXED: Ensures all elements are resized to the target_size (original image size).
        """
        cam = cam_tensor.squeeze().detach().cpu().numpy()
        cam = cv2.resize(cam, target_size) # Resize CAM to original size
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) # Normalize CAM

        # Denormalize and transpose the input tensor back to a standard image array (H, W, C)
        img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)

        # Resize the image array to the target size (original size)
        # This fixes the broadcasting error (ValueError: shapes (256,256,3) (380,380,3)
        img_resized = cv2.resize(img, target_size) 

        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        # Overlay
        overlay = 0.4 * heatmap + 0.6 * img_resized
        return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def _encode_img_to_b64(img_uint8):
        """Encodes uint8 image array to base64 string."""
        _, buf = cv2.imencode(".png", cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode("utf-8")

    def _call_gemini_with_heatmap(self, pred_text: str, overlay_base64: str) -> str:
        """Calls Gemini to explain the heatmap."""
        if not self.has_gemini:
            return f"Prediction: {pred_text} (Gemini API not configured for explanation)"

        # Use a more explicit prompt for better results
        prompt = f"""
        The AI model(s) analyzed an image and predicted it is {pred_text} with high confidence.
        The attached image is a heatmap overlay, where the RED/YELLOW areas show the regions 
        the model focused on to make its prediction.
        
        Based ONLY on the heatmap, explain in simple words (2-3 sentences) why the image 
        is likely {pred_text.lower()}. For example, if the face is red and the prediction is 'Fake', 
        explain that the model noticed artifacts or inconsistencies in the face region.
        """
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(
                [
                    prompt,
                    {
                        "mime_type": "image/png",
                        # Decode the base64 string for the API call
                        "data": base64.b64decode(overlay_base64), 
                    },
                ]
            )
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini call failed: {e}")
            return f"Prediction: {pred_text} (could not generate explanation)"

    # =====================================================
    # ✅ Prediction (Single Model) - FIXED
    # =====================================================
    # =====================================================
    # ✅ Prediction (Single Model) - FINAL FIX
    # =====================================================
    def predict_single(self, model_name: str, image: Image.Image):
        if model_name not in self.models_store:
            raise ValueError(f"Model '{model_name}' not loaded.")

        m = self.models_store[model_name]
        cam_ex = self.cam_extractors[model_name]
        preprocess = self.preprocess_store[model_name]
        original_size = image.size  # Capture original size for correct overlay

        # Ensure input requires grad for GradCAM
        inp = preprocess(image).unsqueeze(0).to(self.device)
        inp.requires_grad_(True)
        out = m(inp)

        # --- Probability Logic ---
        if model_name == "xceptionnet":
            probs_array = out.detach().cpu().numpy().squeeze()
        else:
            probs_array = F.softmax(out, dim=1).detach().cpu().numpy().squeeze()

        if model_name in ["inceptionresnet", "efficientnetb4"]:
            # These are currently np.float32
            probs_fake = probs_array[1]
            probs_real = probs_array[0]
        else:
            # These are currently np.float32
            probs_fake = probs_array[0]
            probs_real = probs_array[1]

        unified_probs = np.array([probs_fake, probs_real])
        pred_idx = int(np.argmax(unified_probs))
        label = "Real" if pred_idx == 1 else "Fake"

        # NOTE: confidence is already cast below.

        # --- CAM Index Logic ---
        if label == "Real":
            original_model_idx = (
                0 if model_name in ["inceptionresnet", "efficientnetb4"] else 1
            )
        else:
            original_model_idx = (
                1 if model_name in ["inceptionresnet", "efficientnetb4"] else 0
            )

        # Generate CAM and Overlay
        cam_map = cam_ex(original_model_idx, out)[0]
        overlay_uint8 = self._get_cam_overlay_and_normalize(
            inp[0], cam_map, original_size
        )
        overlay_b64 = self._encode_img_to_b64(overlay_uint8)

        # Generate Explanation
        if label == "Fake":
            explanation = self._call_gemini_with_heatmap(label, overlay_b64)
        else:
            explanation = "The model found consistent texture and normal details across the image, which suggests it is likely a real photograph."

        return {
            "model": model_name,
            "prediction": label,
            "confidence": float(unified_probs[pred_idx]),  # Explicit cast
            "probabilities": {
                "real": float(probs_real),  # <-- FIX IS HERE
                "fake": float(probs_fake),  # <-- FIX IS HERE
            },
            "heatmap": overlay_b64,
            "explanation": explanation,
        }

    # =====================================================
    # ✅ Prediction (Ensemble) - FIXED
    # =====================================================
    def predict_ensemble(self, image: Image.Image):
        per_model_probs = {}
        all_unified_probs = [] # Collect unified [Fake, Real] probabilities
        cams, weights = [], []
        size = image.size

        for name, m in self.models_store.items():
            preprocess = self.preprocess_store[name]
            cam_ex = self.cam_extractors[name]
            inp = preprocess(image).unsqueeze(0).to(self.device)
            inp.requires_grad_(True) # Important for CAM
            out = m(inp)

            # --- Probability & Inversion Logic ---
            if name == "xceptionnet":
                probs_array = out.detach().cpu().numpy().squeeze()
            else:
                probs_array = F.softmax(out, dim=1).detach().cpu().numpy().squeeze()

            if name in ["inceptionresnet", "efficientnetb4"]:
                probs_fake = probs_array[1] 
                probs_real = probs_array[0] 
                # Model's 'Fake' index is 1, Model's 'Real' index is 0
            else:
                probs_fake = probs_array[0]
                probs_real = probs_array[1]
                # Model's 'Fake' index is 0, Model's 'Real' index is 1

            unified_probs = np.array([probs_fake, probs_real]) # Always [Fake, Real]
            all_unified_probs.append(unified_probs)
            pred_idx = int(np.argmax(unified_probs))
            label = "Real" if pred_idx == 1 else "Fake"

            # --- CAM Index Logic for Ensemble CAM Weighting ---
            if label == "Real":
                original_model_idx = 0 if name in ["inceptionresnet", "efficientnetb4"] else 1
            else:
                original_model_idx = 1 if name in ["inceptionresnet", "efficientnetb4"] else 0

            per_model_probs[name] = {"real": float(probs_real), "fake": float(probs_fake)}

            # Generate CAM
            cam = cam_ex(original_model_idx, out)[0].squeeze().cpu().numpy()
            cam = cv2.resize(cam, size)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cams.append(cam)
            weights.append(unified_probs[pred_idx]) # Use confidence from the unified array

        # --- Ensemble Aggregation ---
        cams = np.stack(cams)
        weights = np.array(weights)
        weights /= weights.sum() + 1e-8

        ensemble_cam = (weights[:, None, None] * cams).sum(0)
        ensemble_cam = (ensemble_cam - ensemble_cam.min()) / (
            ensemble_cam.max() - ensemble_cam.min() + 1e-8
        )

        avg_unified_probs = np.mean(np.stack(all_unified_probs), axis=0) # Average the [Fake, Real] arrays
        pred_idx = int(np.argmax(avg_unified_probs))
        ensemble_label = "Real" if pred_idx == 1 else "Fake"

        # --- Final Output Formatting ---
        heat = np.uint8(255 * ensemble_cam)
        heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        img = np.array(image).astype(np.float32) / 255.0
        overlay = 0.4 * heatmap + 0.6 * img
        overlay_uint8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
        overlay_b64 = self._encode_img_to_b64(overlay_uint8)

        # Generate Explanation
        if ensemble_label == "Fake":
            explanation = self._call_gemini_with_heatmap(ensemble_label, overlay_b64)
        else:
            explanation = "The ensemble model found consistent texture and normal details across the image, suggesting it is likely a real photograph."

        return {
            "ensemble_prediction": ensemble_label,
            "ensemble_confidence": float(avg_unified_probs[pred_idx]),
            "ensemble_probabilities": {
                "real": float(avg_unified_probs[1]),
                "fake": float(avg_unified_probs[0]),
            },
            "per_model_confidences": per_model_probs,
            "heatmap": overlay_b64,
            "explanation": explanation, # NEW
        }
