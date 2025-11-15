import os
from io import BytesIO
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "model/gender_cnn.h5")
FACE_CASCADE_PATH = os.getenv("FACE_CASCADE_PATH", "model/haarcascade_frontalface_default.xml")
CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
)

# Lazy-loaded modules/objects
_cv2 = None
_tf = None
_face_cascade = None
_model = None


def _ensure_dirs():
    os.makedirs(os.path.dirname(FACE_CASCADE_PATH), exist_ok=True)


def _ensure_haarcascade():
    _ensure_dirs()
    if not os.path.exists(FACE_CASCADE_PATH):
        try:
            r = requests.get(CASCADE_URL, timeout=10)
            r.raise_for_status()
            with open(FACE_CASCADE_PATH, "wb") as f:
                f.write(r.content)
        except Exception:
            # If download fails, we'll error later when loading
            pass


def _load_opencv():
    global _cv2
    if _cv2 is None:
        try:
            import cv2  # type: ignore
        except Exception as e:
            raise RuntimeError(
                f"OpenCV not available: {e}. Ensure opencv-python-headless is installed."
            )
        _cv2 = cv2
    return _cv2


def _load_tensorflow():
    global _tf
    if _tf is None:
        try:
            import tensorflow as tf  # type: ignore
        except Exception as e:
            raise RuntimeError(
                f"TensorFlow not available: {e}. The server runs without it, but prediction requires TensorFlow."
            )
        _tf = tf
    return _tf


def load_dependencies():
    global _face_cascade, _model

    cv2 = _load_opencv()

    # Load face cascade
    if _face_cascade is None:
        _ensure_haarcascade()
        if not os.path.exists(FACE_CASCADE_PATH):
            raise RuntimeError(
                "Face cascade file not found and could not be downloaded. "
                "Please add haarcascade_frontalface_default.xml into the model/ folder."
            )
        fc = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        if fc.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier.")
        _face_cascade = fc

    # Load model lazily with TF only when needed
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Model file not found. Place your trained model at model/gender_cnn.h5 or set MODEL_PATH."
            )
        tf = _load_tensorflow()
        try:
            _model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


class FaceResult(BaseModel):
    label: str
    confidence: float
    bbox: list


@app.get("/")
async def root():
    return {"message": "Gender prediction API is running"}


@app.get("/status")
async def status():
    status = {
        "model_path": MODEL_PATH,
        "model_available": os.path.exists(MODEL_PATH),
        "cascade_path": FACE_CASCADE_PATH,
        "cascade_available": os.path.exists(FACE_CASCADE_PATH),
        "tensorflow_importable": False,
        "opencv_importable": False,
    }
    # Try lazy checks without failing app
    try:
        _load_tensorflow()
        status["tensorflow_importable"] = True
    except Exception:
        status["tensorflow_importable"] = False
    try:
        _load_opencv()
        status["opencv_importable"] = True
    except Exception:
        status["opencv_importable"] = False
    return JSONResponse(status)


@app.post("/predict")
async def predict_gender(file: UploadFile = File(...)):
    """Accepts an image, detects faces, and predicts gender for each face."""
    try:
        load_dependencies()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    cv2 = _load_opencv()
    tf = _load_tensorflow()  # ensure loaded

    # Read once
    contents = await file.read()

    # Try PIL first
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
    except Exception:
        # Fallback to OpenCV decode
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    results = []
    for (x, y, w, h) in faces:
        face_img = img_np[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (128, 128))
        face_norm = face_resized / 255.0
        face_batch = np.expand_dims(face_norm, axis=0)
        pred = float(_model.predict(face_batch, verbose=0)[0][0])
        label = "Male" if pred >= 0.5 else "Female"
        conf = pred if label == "Male" else 1.0 - pred
        results.append({"label": label, "confidence": round(conf, 3), "bbox": [int(x), int(y), int(w), int(h)]})

    return {"faces": results}


@app.post("/predict/annotated")
async def predict_annotated(file: UploadFile = File(...)):
    try:
        load_dependencies()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    cv2 = _load_opencv()

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face_img = img_rgb[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (128, 128))
        face_norm = face_resized / 255.0
        face_batch = np.expand_dims(face_norm, axis=0)
        pred = float(_model.predict(face_batch, verbose=0)[0][0])
        label = "Male" if pred >= 0.5 else "Female"
        conf = pred if label == "Male" else 1.0 - pred
        color = (0, 200, 0) if label == "Male" else (200, 0, 150)
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img_rgb, f"{label} {conf:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    annotated_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode('.jpg', annotated_bgr)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
