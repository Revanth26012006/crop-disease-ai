import onnxruntime as ort
import numpy as np
import cv2

# Load ONNX model
session = ort.InferenceSession(
    "runs/train/exp2/weights/best.onnx",
    providers=["CPUExecutionProvider"]
)

def preprocess(im0):
    img = cv2.resize(im0, (640, 640))
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def run_onnx(im0):
    img = preprocess(im0)
    outputs = session.run(None, {"images": img})
    return outputs