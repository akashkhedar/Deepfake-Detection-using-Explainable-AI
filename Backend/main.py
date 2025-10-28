# backend/main.py

import logging
import os
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image

# Import service module robustly so running `uvicorn main:app` from the Backend
# folder or running as a package both work.
try:
    from .service import ModelManager
except Exception:
    # fallback to absolute import when module is executed as a script
    from service import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- APP & CORS ---
app = FastAPI()
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

# create manager (models are loaded on startup)
manager = ModelManager()


@app.on_event("startup")
def startup_event():
    # read optional env var LOAD_MODELS (comma separated names or 'all')
    load_cfg = os.getenv("LOAD_MODELS", "all")
    if load_cfg.strip().lower() == "all":
        names = None
    else:
        names = [n.strip().lower() for n in load_cfg.split(",") if n.strip()]
    try:
        manager.load_models(names=names)
    except Exception as e:
        logger.exception(f"Error loading models on startup: {e}")


@app.get("/")
async def root():
    return {
        "message": "DeepFake Detection API (ensemble + individual models) is running"
    }


@app.get("/models/")
async def get_models():
    return {"available_models": manager.get_available_models()}


@app.get("/status/")
async def get_status():
    return manager.get_status()


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    model: str = Query(
        None, description="Model name (if omitted, ensemble will be used)"
    ),
):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        if model:
            model = model.lower()
            if model not in manager.get_available_models():
                return JSONResponse(
                    content={"error": f"Model {model} not found."}, status_code=400
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
        logger.exception(f"Error in prediction: {e}")
        return JSONResponse(
            content={"error": f"Prediction failed: {str(e)}"},
            status_code=500,
            headers={"Access-Control-Allow-Origin": "*"},
        )
