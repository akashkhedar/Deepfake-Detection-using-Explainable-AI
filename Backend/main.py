import logging
import os
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

try:
    from .service import ModelManager
except ImportError:
    from service import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("service")

app = FastAPI(title="DeepFake Detection API", version="1.0")

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

# Global Model Manager
manager = ModelManager()


@app.on_event("startup")
def startup_event():
    """Load models automatically on API startup."""
    load_cfg = os.getenv("LOAD_MODELS", "all")

    if load_cfg.strip().lower() == "all":
        names = None
    else:
        names = [n.strip().lower() for n in load_cfg.split(",") if n.strip()]

    try:
        manager.load_models(names=names)
        logger.info("✅ Models loaded successfully on startup.")
    except Exception as e:
        logger.exception(f"❌ Error loading models on startup: {e}")


# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "DeepFake Detection API (ensemble + individual models) is running"
    }


@app.get("/models/")
async def get_models():
    """Return list of loaded models."""
    return {"available_models": manager.get_loaded_models()}


@app.get("/status/")
async def get_status():
    """Return system and model loading status."""
    return manager.get_status()


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    model: str = Query(
        None, description="Model name (optional). If omitted, ensemble will be used."
    ),
):
    """Run prediction using a single model or ensemble."""
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")

        if model:
            model = model.lower()
            if model not in manager.get_loaded_models():
                return JSONResponse(
                    content={"error": f"Model '{model}' not found."},
                    status_code=400,
                )
            resp = manager.predict_single(model, image)
            return JSONResponse(
                content=resp, headers={"Access-Control-Allow-Origin": "*"}
            )
        else:
            resp = manager.predict_ensemble(image)
            return JSONResponse(
                content=resp, headers={"Access-Control-Allow-Origin": "*"}
            )

    except Exception as e:
        logger.exception(f"❌ Error during prediction: {e}")
        return JSONResponse(
            content={"error": f"Prediction failed: {str(e)}"},
            status_code=500,
            headers={"Access-Control-Allow-Origin": "*"},
        )
