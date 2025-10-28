import os
import io
import base64
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms, models

from dotenv import load_dotenv
import google.generativeai as genai

# GradCAM
from torchcam.methods import GradCAM

load_dotenv()
logger = logging.getLogger(__name__)

# --- CONFIG ---
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# default checkpoint paths (relative to Backend/)
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


class ModelManager:
    """Loads models (optionally lazily), runs inference and builds CAM overlays.

    Usage:
      manager = ModelManager()
      manager.load_models()  # loads available models based on CHECKPOINTS
      manager.get_available_models()
      result = manager.predict(image_pil, model=None)  # ensemble
    """

    def __init__(self, device: str = DEVICE, checkpoints: dict = CHECKPOINTS):
        self.device = device
        self.checkpoints = checkpoints
        self.models_store: Dict[str, torch.nn.Module] = {}
        self.cam_extractors: Dict[str, GradCAM] = {}
        self.preprocess_store: Dict[str, transforms.Compose] = {}
        self.model_status: Dict[str, dict] = {}
        # configure Gemini if key present
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                logger.info("Configured Gemini API")
            except Exception as e:
                logger.warning(f"Failed to configure Gemini: {e}")

    # ---- Model factory & loader ----
    def build_model(
        self, name: str, num_classes: int = 2
    ) -> Tuple[torch.nn.Module, object]:
        name = name.lower()
        if name == "resnet50":
            m = models.resnet50(weights=None)
            num_ftrs = m.fc.in_features
            m.fc = torch.nn.Linear(num_ftrs, num_classes)
            target_layer = m.layer4[-1].conv2
        elif name == "resnet152":
            m = models.resnet152(weights=None)
            num_ftrs = m.fc.in_features
            m.fc = torch.nn.Linear(num_ftrs, num_classes)
            target_layer = m.layer4[-1].conv2
        elif name == "inceptionresnet":
            from torchvision.models import inception_v3

            m = inception_v3(weights=None, aux_logits=False)
            num_ftrs = m.fc.in_features
            m.fc = torch.nn.Linear(num_ftrs, num_classes)
            target_layer = getattr(m, "Mixed_7c", None)
        elif name == "xceptionnet":
            try:
                from models.xception import xception

                m = xception(num_classes=num_classes)
            except Exception as e:
                raise RuntimeError("Xception model loader not found: %s" % e)
            target_layer = getattr(m, "block4", None) or getattr(m, "layer4", None)
        elif name == "efficientnetb4":
            try:
                m = models.efficientnet_b4(weights=None)
                num_ftrs = m.classifier[1].in_features
                m.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
                target_layer = m.features[-1]
            except Exception:
                raise RuntimeError(
                    "EfficientNet-B4 not available in torchvision; adapt to your implementation."
                )
        else:
            raise ValueError(f"Unknown model name: {name}")
        return m, target_layer

    def load_checkpoint(
        self, m: torch.nn.Module, checkpoint_path: str, device: str = None
    ):
        device = device or self.device
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        # Try loading the checkpoint. Newer PyTorch versions may restrict
        # classes allowed when loading weights_only files; attempt a few
        # fallbacks if the first load fails.
        try:
            ck = torch.load(checkpoint_path, map_location=device)
        except Exception as e:
            logger.warning(f"Initial torch.load failed for {checkpoint_path}: {e}")
            # Try to relax safe globals (if available) or explicitly request
            # weights_only=False when supported. These fallbacks may be
            # necessary for older checkpoints saved with arbitrary objects.
            tried = False
            # 1) Try safe_globals context manager if available
            try:
                if hasattr(torch.serialization, "safe_globals"):
                    # allowlist numpy scalar global if present
                    try:
                        ng = [np.core.multiarray.scalar]
                        with torch.serialization.safe_globals(ng):
                            ck = torch.load(checkpoint_path, map_location=device)
                            tried = True
                    except Exception:
                        # fall through to other strategies
                        pass
            except Exception:
                pass

            # 2) Try add_safe_globals (older API)
            if not tried:
                try:
                    if hasattr(torch.serialization, "add_safe_globals"):
                        try:
                            torch.serialization.add_safe_globals(
                                [np.core.multiarray.scalar]
                            )
                            ck = torch.load(checkpoint_path, map_location=device)
                            tried = True
                        except Exception:
                            pass
                except Exception:
                    pass

            # 3) As a last resort, attempt to load with weights_only=False (if supported)
            if not tried:
                try:
                    ck = torch.load(
                        checkpoint_path, map_location=device, weights_only=False
                    )
                    tried = True
                except TypeError:
                    # weights_only arg not supported on this torch version; try plain load again
                    try:
                        ck = torch.load(checkpoint_path, map_location=device)
                        tried = True
                    except Exception as e2:
                        logger.exception(f"Fallback torch.load attempts failed: {e2}")
                        raise
                except Exception as e2:
                    logger.exception(f"Fallback torch.load attempts failed: {e2}")
                    raise

        # Determine state dict from checkpoint formats
        if isinstance(ck, dict):
            if "model_state_dict" in ck:
                state = ck["model_state_dict"]
            elif "state_dict" in ck:
                state = ck["state_dict"]
            else:
                state = ck
        else:
            state = ck
        new_state = {}
        for k, v in state.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state[new_key] = v
        m.load_state_dict(new_state, strict=False)
        return m

    def load_models(
        self, names: Optional[List[str]] = None, to_device: Optional[str] = None
    ):
        """Load a set of models. By default loads all checkpoints listed in self.checkpoints."""
        to_device = to_device or self.device
        names_to_load = names or list(self.checkpoints.keys())
        for name in names_to_load:
            status = {
                "checkpoint": self.checkpoints.get(name),
                "built": False,
                "checkpoint_loaded": False,
                "moved_to_device": False,
                "cam_ready": False,
                "available": False,
                "errors": [],
            }
            ckpt = self.checkpoints.get(name)
            if not ckpt:
                msg = f"No checkpoint configured for {name}, skipping"
                logger.warning(msg)
                status["errors"].append(msg)
                self.model_status[name] = status
                continue
            try:
                m, target_layer = self.build_model(name)
                status["built"] = True
            except Exception as e:
                msg = f"Could not build model {name}: {e}"
                logger.warning(msg)
                status["errors"].append(str(e))
                self.model_status[name] = status
                continue
            try:
                m = self.load_checkpoint(m, ckpt, device=to_device)
                status["checkpoint_loaded"] = True
            except Exception as e:
                msg = f"Loading checkpoint failed for {name}: {e}"
                logger.warning(msg)
                status["errors"].append(str(e))
                # continue with uninitialized weights if needed
            m = m.to(to_device)
            status["moved_to_device"] = True
            m.eval()
            for p in m.parameters():
                p.requires_grad_(True)
            if target_layer is None:
                msg = f"Target layer for GradCAM not set for {name}; skipping CAM for this model"
                logger.warning(msg)
                status["errors"].append("No target_layer for CAM")
                self.model_status[name] = status
                continue
            try:
                cam = GradCAM(m, target_layer=target_layer)
                status["cam_ready"] = True
            except Exception as e:
                msg = f"Failed to create GradCAM extractor for {name}: {e}"
                logger.warning(msg)
                status["errors"].append(str(e))
                self.model_status[name] = status
                continue
            self.models_store[name] = m
            self.cam_extractors[name] = cam
            self.preprocess_store[name] = make_preprocess(IMG_SIZE_MAP.get(name, 224))
            status["available"] = True
            self.model_status[name] = status
            logger.info(f"Loaded model: {name}")

    def get_available_models(self) -> List[str]:
        return list(self.models_store.keys())

    def get_status(self) -> dict:
        return {
            "device": self.device,
            "models": self.model_status,
            "available_models": self.get_available_models(),
        }

    # ---- Utilities for CAM & overlay ----
    @staticmethod
    def encode_overlay_to_base64(overlay_rgb_uint8: np.ndarray) -> str:
        overlay_bgr = cv2.cvtColor(overlay_rgb_uint8, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(
            ".png", overlay_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6]
        )
        if not success:
            raise RuntimeError("Failed to encode image")
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def get_cam_overlay_for_single_model(
        img_pil: Image.Image,
        img_tensor: torch.Tensor,
        cam_map: torch.Tensor,
        original_size,
    ):
        cam_np = cam_map.squeeze(0).detach().cpu().numpy().astype(np.float32)
        cam_resized = cv2.resize(cam_np, original_size, interpolation=cv2.INTER_LINEAR)
        cam_norm = (cam_resized - cam_resized.min()) / (
            cam_resized.max() - cam_resized.min() + 1e-8
        )
        heatmap = np.uint8(255 * cam_norm)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        img_norm = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_denorm = np.clip(std * img_norm + mean, 0, 1)
        img_uint8 = (img_denorm * 255).astype(np.uint8)
        img_resized = cv2.resize(
            img_uint8, original_size, interpolation=cv2.INTER_LINEAR
        )

        overlay = 0.4 * heatmap + 0.6 * (img_resized.astype(np.float32) / 255.0)
        overlay = np.clip(overlay, 0, 1)
        overlay_uint8 = (overlay * 255).astype(np.uint8)
        return overlay_uint8, cam_norm

    @staticmethod
    def build_ensembled_heatmap(
        per_model_cam_resized: Dict[str, np.ndarray], per_model_probs: Dict[str, float]
    ):
        weights = np.array(
            [per_model_probs.get(n, 0.0) for n in per_model_cam_resized.keys()],
            dtype=np.float32,
        )
        total = weights.sum() + 1e-8
        norm_weights = weights / total
        cams = np.stack(
            [per_model_cam_resized[n] for n in per_model_cam_resized.keys()], axis=0
        )
        weighted = (norm_weights[:, None, None] * cams).sum(axis=0)
        weighted = (weighted - weighted.min()) / (
            weighted.max() - weighted.min() + 1e-8
        )
        return weighted

    # ---- Gemini explanation ----
    def call_gemini_with_heatmap(self, pred_text: str, overlay_base64: str) -> str:
        prompt = f"""
        The AI model ensemble analyzed an image and predicted it is {pred_text}.
        Here is the heatmap showing which areas of the image the model(s) focused on.
        Please explain in simple words (2-3 sentences) why the image is likely {pred_text.lower()}.
        """
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(
                [
                    prompt,
                    {
                        "mime_type": "image/png",
                        "data": base64.b64decode(overlay_base64),
                    },
                ]
            )
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini call failed: {e}")
            return f"Prediction: {pred_text} (could not generate explanation)"

    # ---- Prediction API ----
    def predict_single(self, model: str, image: Image.Image) -> dict:
        if model not in self.models_store:
            raise ValueError(f"Model {model} not loaded")
        m = self.models_store[model]
        cam_extractor = self.cam_extractors.get(model)
        preprocess = self.preprocess_store.get(model)
        original_size = image.size
        inp = preprocess(image).unsqueeze(0).to(self.device)
        inp.requires_grad_(True)
        # Important: do NOT disable grads here; GradCAM needs gradients on outputs
        out = m(inp)
        probs = F.softmax(out, dim=1).detach().cpu().numpy().squeeze(0)
        pred_idx = int(np.argmax(probs))
        pred_label = "Real" if pred_idx == 1 else "Fake"

        cam_tensor = cam_extractor(pred_idx, out)[0]
        overlay_uint8, cam_norm = self.get_cam_overlay_for_single_model(
            image, inp[0].cpu(), cam_tensor, original_size
        )
        overlay_b64 = self.encode_overlay_to_base64(overlay_uint8)

        if pred_label == "Fake":
            explanation = self.call_gemini_with_heatmap(pred_label, overlay_b64)
        else:
            explanation = "The image appears natural with consistent details, so it is likely real."

        return {
            "model": model,
            "prediction": pred_label,
            "confidence": float(probs[pred_idx]),
            "probabilities": {"real": float(probs[1]), "fake": float(probs[0])},
            "heatmap": overlay_b64,
            "explanation": explanation,
        }

    def predict_ensemble(self, image: Image.Image) -> dict:
        per_model_probs = {}
        per_model_outs = {}
        per_model_cams_resized = {}
        original_size = image.size

        for name, m in self.models_store.items():
            preprocess = self.preprocess_store[name]
            cam_ext = self.cam_extractors[name]
            inp = preprocess(image).unsqueeze(0).to(self.device)
            inp.requires_grad_(True)
            # Important: do NOT disable grads; GradCAM needs gradients on outputs
            out = m(inp)
            probs = F.softmax(out, dim=1).detach().cpu().numpy().squeeze(0)
            per_model_outs[name] = probs
            # Compute CAMs for both classes; retain the graph on first backward
            # so the second backward can run without error.
            cam0 = cam_ext(0, out, retain_graph=True)[0]
            cam1 = cam_ext(1, out)[0]
            cam0_np = cv2.resize(
                cam0.squeeze(0).detach().cpu().numpy().astype(np.float32),
                original_size,
                interpolation=cv2.INTER_LINEAR,
            )
            cam0_np = (cam0_np - cam0_np.min()) / (cam0_np.max() - cam0_np.min() + 1e-8)
            cam1_np = cv2.resize(
                cam1.squeeze(0).detach().cpu().numpy().astype(np.float32),
                original_size,
                interpolation=cv2.INTER_LINEAR,
            )
            cam1_np = (cam1_np - cam1_np.min()) / (cam1_np.max() - cam1_np.min() + 1e-8)
            per_model_cams_resized[name] = {"0": cam0_np, "1": cam1_np}
            per_model_probs[name] = {"0": float(probs[0]), "1": float(probs[1])}

        model_names = list(per_model_outs.keys())
        all_probs = np.stack([per_model_outs[n] for n in model_names], axis=0)
        ensemble_probs = all_probs.mean(axis=0)
        ensemble_pred_idx = int(np.argmax(ensemble_probs))
        ensemble_label = "Real" if ensemble_pred_idx == 1 else "Fake"

        per_model_cam_for_ens = {
            n: per_model_cams_resized[n][str(ensemble_pred_idx)] for n in model_names
        }
        per_model_prob_for_ens = {
            n: per_model_probs[n][str(ensemble_pred_idx)] for n in model_names
        }
        ensembled_cam = self.build_ensembled_heatmap(
            per_model_cam_for_ens, per_model_prob_for_ens
        )

        img_np = np.array(image.resize(original_size))
        heat_uint8 = np.uint8(255 * ensembled_cam)
        heatmap = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_float = img_np.astype(np.float32) / 255.0
        overlay = 0.4 * heatmap + 0.6 * img_float
        overlay_uint8 = np.clip(overlay, 0, 1)
        overlay_uint8 = (overlay_uint8 * 255).astype(np.uint8)
        overlay_b64 = self.encode_overlay_to_base64(overlay_uint8)

        if ensemble_label == "Fake":
            explanation = self.call_gemini_with_heatmap(ensemble_label, overlay_b64)
        else:
            explanation = "The image appears natural with consistent details, so it is likely real."

        per_model_confidences = {
            n: {"fake": per_model_probs[n]["0"], "real": per_model_probs[n]["1"]}
            for n in model_names
        }
        response = {
            "ensemble_prediction": ensemble_label,
            "ensemble_confidence": float(ensemble_probs[ensemble_pred_idx]),
            "ensemble_probabilities": {
                "real": float(ensemble_probs[1]),
                "fake": float(ensemble_probs[0]),
            },
            "per_model_confidences": per_model_confidences,
            "heatmap": overlay_b64,
            "explanation": explanation,
        }
        return response
