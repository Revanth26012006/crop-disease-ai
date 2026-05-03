from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import requests
import torch
import onnxruntime as ort
import os

from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

app = FastAPI()

# -------------------------------
# LOAD ONNX MODEL
# -------------------------------
MODEL_PATH = "best.onnx"

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy"
]

stride = 32


# -------------------------------
# PREPROCESS
# -------------------------------
def preprocess(file):
    file.seek(0)
    file_bytes = np.frombuffer(file.read(), np.uint8)
    im0 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if im0 is None:
        raise ValueError("Invalid image file")

    im, _, _ = letterbox(im0, (640, 640), auto=True, stride=stride)
    im = im[:, :, ::-1].transpose(2, 0, 1)
    im = np.ascontiguousarray(im)

    im = im.astype(np.float32) / 255.0
    im = np.expand_dims(im, 0)

    return im, im0


# -------------------------------
# LLM (AUTO SWITCH)
# -------------------------------
def generate_advisory(disease_name):
    try:
        # 🔥 Clean disease name
        clean_name = disease_name.replace(" (Low Confidence)", "").replace("Unknown", "plant disease")

        prompt = f"""
You are an agricultural expert.

Explain the plant disease: {clean_name}

Include:
- What it is
- Causes
- Treatment
- Prevention

Give direct practical advice.
"""

        # 🔥 Detect environment
        ollama_url = os.getenv("OLLAMA_URL")

        # -------------------------------
        # LOCAL DOCKER (OLLAMA AVAILABLE)
        # -------------------------------
        if ollama_url:
            res = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": "phi", "prompt": prompt, "stream": False},
                timeout=20
            )
            return res.json().get("response", "No advisory generated")

        # -------------------------------
        # DEPLOYMENT (NO OLLAMA)
        # -------------------------------
        else:
            return f"""
{clean_name} is a plant disease that affects crop health.

General advice:
- Remove infected leaves
- Avoid overwatering
- Use appropriate fungicides
- Maintain plant hygiene
"""

    except:
        return "Advisory unavailable"


# -------------------------------
# PREDICT
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        im, im0 = preprocess(file.file)

        # ONNX inference
        outputs = session.run(None, {input_name: im})

        if outputs is None or len(outputs) == 0:
            return {
                "detections": [],
                "advisory": "Model returned no output"
            }

        raw_pred = torch.tensor(outputs[0])

        pred = non_max_suppression(raw_pred, conf_thres=0.01, iou_thres=0.45)

        # -------------------------------
        # NO DETECTION
        # -------------------------------
        if pred is None or len(pred) == 0 or len(pred[0]) == 0:
            return {
                "detections": [{
                    "class": "No disease detected",
                    "confidence": 0.0,
                    "bbox": [0, 0, 0, 0]
                }],
                "advisory": "No clear disease detected. Try a clearer image."
            }

        # -------------------------------
        # VALID DETECTION
        # -------------------------------
        det = pred[0]
        det[:, :4] = scale_boxes((640, 640), det[:, :4], im0.shape)

        top = det[det[:, 4].argmax()]
        x1, y1, x2, y2, conf, cls = top.tolist()

        cls = int(cls)
        disease = names[cls] if cls < len(names) else "Unknown"

        confidence = float(conf)

        # 🔥 Better UX
        if confidence < 0.05:
            disease = f"{disease} (Low Confidence)"

        result = {
            "class": disease,
            "confidence": confidence,
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        }

        advisory = generate_advisory(disease)

        return {
            "detections": [result],
            "advisory": advisory
        }

    except Exception as e:
        return {
            "detections": [],
            "advisory": "Error occurred",
            "error": str(e)
        }