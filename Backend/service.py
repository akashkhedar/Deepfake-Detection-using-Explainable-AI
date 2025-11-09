import os
import base64
import logging
from typing import Dict, List, Optional, Tuple

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

# Global Config
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
load_dotenv()
logger = logging.getLogger("service")
logger.setLevel(logging.INFO)

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

MODEL_SCHEMA = {
    "resnet50": {"type": "softmax", "order": ["real", "fake"], "sharpen": 1.0},
    "resnet152": {"type": "softmax", "order": ["real", "fake"], "sharpen": 1.0},
    "inceptionresnet": {"type": "softmax", "order": ["real", "fake"], "sharpen": 1.0},
    "xceptionnet": {"type": "sigmoid", "pos_label": "fake", "sharpen": 8.0},
    "efficientnetb4": {"type": "softmax", "order": ["real", "fake"], "sharpen": 2.0},
}


def make_preprocess(size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# Model Manager
class ModelManager:
    def __init__(self, device: str = DEVICE, checkpoints: dict = CHECKPOINTS):
        self.device = device
        self.checkpoints = checkpoints
        self.models_store: Dict[str, torch.nn.Module] = {}
        self.cam_extractors: Dict[str, GradCAM] = {}
        self.preprocess_store: Dict[str, transforms.Compose] = {}
        self.model_status: Dict[str, dict] = {}
        self.has_gemini = False

        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.has_gemini = True
                logger.info("✅ Gemini API configured successfully.")
            except Exception as e:
                logger.warning(f"⚠️ Gemini configuration failed: {e}")

    # Model Factory
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
            target_layer = m.conv2d_7b

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
            target_layer = m.conv4

        elif name == "efficientnetb4":
            m = timm.create_model(
                "efficientnet_b4", pretrained=False, num_classes=num_classes
            )
            target_layer = m.blocks[-1][-1].conv_pwl

        else:
            raise ValueError(f"Unknown model: {name}")

        return m, target_layer

    # Explain Prediction with Gemini (XAI Text)
    def explain_prediction(self, image_b64: str, label: str, confidence: float) -> str:
        """Generate a natural-language explanation using Gemini Vision."""
        if not self.has_gemini:
            return "Gemini API key not configured. Explanation unavailable."

        try:
            img_bytes = base64.b64decode(image_b64)
            img_obj = {
                "mime_type": "image/png",
                "data": img_bytes,
            }

            prompt = f"Explain briefly (2-3 sentences) why the model predicted the image as {label.lower()} based on the attached heatmap overlay."

            model = genai.GenerativeModel("gemini-2.5-flash")
            resp = model.generate_content([prompt, img_obj])
            return resp.text.strip()
        except Exception as e:
            logger.warning(f"⚠️ Gemini explanation failed: {e}")
            return "Explanation could not be generated."

    # Safe Checkpoint Loader
    def load_checkpoint(self, model: torch.nn.Module, path: str, device: str = None):
        device = device or self.device
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Always load once, safely
        with torch.serialization.safe_globals([np.core.multiarray.scalar]):
            ckpt = torch.load(path, map_location=device, weights_only=False)

        if isinstance(ckpt, dict):
            state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        else:
            state = ckpt

        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)
        return model

    # Load All Models
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
                for p in m.parameters():
                    p.requires_grad_(True)

                cam = GradCAM(m, target_layer=target_layer)
                self.models_store[name] = m
                self.cam_extractors[name] = cam
                self.preprocess_store[name] = make_preprocess(IMG_SIZE_MAP[name])
                status["available"] = True
                logger.info(f"✅ Loaded model {name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {name}: {e}")
                status["errors"].append(str(e))

            self.model_status[name] = status

    # Helper Functions
    def get_loaded_models(self):
        return list(self.models_store.keys())

    def get_status(self):
        return {
            "device": self.device,
            "models": self.model_status,
            "available_models": self.get_loaded_models(),
        }

    # unified output postprocessing
    def _postprocess_output(self, name: str, out: torch.Tensor):
        schema = MODEL_SCHEMA.get(name, {"type": "softmax", "order": ["real", "fake"]})
        out_det = out.detach().cpu()

        if schema["type"] == "softmax":
            probs = F.softmax(out_det, dim=1).numpy().squeeze()
            order = schema["order"]
            fake = float(probs[1]) if order == ["real", "fake"] else float(probs[0])
            real = float(probs[0]) if order == ["real", "fake"] else float(probs[1])
        else:
            arr = out_det.numpy().squeeze()
            p = float(arr[1]) if arr.ndim > 0 and arr.shape[0] == 2 else float(arr)
            fake = p
            real = 1.0 - p

        unified = np.array([fake, real])

        if name in ["xceptionnet", "resnet152", "resnet50"]:
            unified = unified[::-1]

        sharpen = schema.get("sharpen", 1.0)
        if sharpen != 1.0:
            unified = np.power(unified, sharpen)
            unified /= unified.sum() + 1e-8

        return np.clip(unified, 0, 1)

    def _encode_b64(self, img_uint8):
        _, buf = cv2.imencode(".png", cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode("utf-8")

    def _make_overlay(self, pre_tensor, cam_np, target_size):
        # Resize and normalize CAM
        cam = cv2.resize(cam_np, target_size)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Apply colormap (JET → RGB)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Reconstruct original RGB image
        img = pre_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(
            img * np.array([0.229, 0.224, 0.225], dtype=np.float32)
            + np.array([0.485, 0.456, 0.406], dtype=np.float32),
            0,
            1,
        )
        img = cv2.resize(img, target_size)
        img = (img * 255).astype(np.uint8)  # ✅ ensure same dtype (uint8)

        # ✅ Blend smoothly with correct dtype
        alpha = 0.35  # controls heatmap visibility (0.3–0.4 ideal)
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

        # Optional soft blur for smoother appearance
        overlay = cv2.GaussianBlur(overlay, (7, 7), 0)

        return overlay

    # Single Prediction
    def predict_single(self, model_name: str, image: Image.Image):
        m = self.models_store[model_name]
        cam_ex = self.cam_extractors[model_name]
        preprocess = self.preprocess_store[model_name]
        inp = preprocess(image).unsqueeze(0).to(self.device)
        inp.requires_grad_(True)

        out = m(inp)
        unified = self._postprocess_output(model_name, out)
        label_idx = int(np.argmax(unified))
        label = "Real" if label_idx == 1 else "Fake"

        cam_tensor = cam_ex(label_idx, out)[0]
        overlay = self._make_overlay(
            inp[0], cam_tensor.squeeze().detach().cpu().numpy(), image.size
        )
        overlay_b64 = self._encode_b64(overlay)

        explanation = self.explain_prediction(overlay_b64, label, float(max(unified)))

        return {
            "model": model_name,
            "prediction": label,
            "confidence": float(max(unified)),
            "probabilities": {"real": float(unified[1]), "fake": float(unified[0])},
            "heatmap": overlay_b64,
            "explanation": explanation,
        }

    # Ensemble Prediction
    def predict_ensemble(self, image: Image.Image):
        per_model = {}
        cams, weights, unified_list = [], [], []
        size = image.size
        sample_input_tensor = None

        for name, m in self.models_store.items():
            preprocess = self.preprocess_store[name]
            inp = preprocess(image).unsqueeze(0).to(self.device)
            inp.requires_grad_(True)
            if sample_input_tensor is None:
                sample_input_tensor = inp[0]
            out = m(inp)
            unified = self._postprocess_output(name, out)
            unified_list.append(unified)
            label_idx = int(np.argmax(unified))

            cam_tensor = self.cam_extractors[name](label_idx, out)[0]
            cam = cam_tensor.squeeze().detach().cpu().numpy()
            cam = cv2.resize(cam, size, interpolation=cv2.INTER_LINEAR)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cams.append(cam)
            weights.append(unified[label_idx])
            per_model[name] = {"real": float(unified[1]), "fake": float(unified[0])}

        cams = np.stack(cams)
        weights = np.array(weights)
        weights /= weights.sum() + 1e-8
        ensemble_cam = (weights[:, None, None] * cams).sum(0)
        ensemble_cam = (ensemble_cam - ensemble_cam.min()) / (
            ensemble_cam.max() - ensemble_cam.min() + 1e-8
        )

        avg_unified = np.mean(np.stack(unified_list), axis=0)
        label = "Real" if np.argmax(avg_unified) == 1 else "Fake"

        overlay = self._make_overlay(
            sample_input_tensor, # Use the actual preprocessed image tensor
            ensemble_cam,
            size,
        )
        overlay_b64 = self._encode_b64(overlay)

        explanation = self.explain_prediction(
            overlay_b64, label, float(max(avg_unified))
        )

        return {
            "ensemble_prediction": label,
            "ensemble_confidence": float(max(avg_unified)),
            "ensemble_probabilities": {
                "real": float(avg_unified[1]),
                "fake": float(avg_unified[0]),
            },
            "per_model_confidences": per_model,
            "heatmap": overlay_b64,
            "explanation": explanation,
        }
