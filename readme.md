# 🌱 Crop Disease AI

A real-time plant disease detection and advisory system using Deep Learning, ONNX, FastAPI, and Streamlit.

---

## 🚀 Overview

Crop Disease AI is an end-to-end AI system that detects plant diseases from leaf images and provides actionable recommendations. It combines:

* 🧠 YOLO-based object detection (exported to ONNX)
* ⚡ FastAPI backend for inference
* 🌐 Streamlit frontend for user interaction
* 📦 Docker for deployment
* ☁️ Cloud deployment (Render + Streamlit Cloud)

---

## 🧩 System Architecture

```
User (Browser)
   ↓
Streamlit UI (Frontend)
   ↓
FastAPI (Backend API)
   ↓
ONNX Model (Inference)
   ↓
Prediction + Advisory
```

---

## ✨ Features

* 📷 Upload plant leaf images
* 🔍 Real-time disease detection
* 📦 Bounding box visualization
* 📊 Confidence score display
* ⚠️ Low-confidence handling
* 💡 Disease advisory (fallback-based / LLM-ready)
* 🔗 REST API support
* ☁️ Cloud deployment ready

---

## 🛠️ Tech Stack

* **Model:** YOLOv5 (exported to ONNX)
* **Backend:** FastAPI
* **Frontend:** Streamlit
* **Inference:** ONNX Runtime
* **Libraries:** OpenCV, NumPy, Pillow
* **Deployment:** Docker, Render, Streamlit Cloud

---

## 📂 Project Structure

```
crop-disease-ai/
│
├── app.py               # FastAPI backend
├── ui.py                # Streamlit frontend
├── best.onnx            # Trained model
├── Dockerfile           # Container setup
├── requirements.txt     # Dependencies (frontend)
├── utils/               # YOLO utility functions
└── README.md
```

---

## ⚙️ Installation (Local Setup)

### 1. Clone the repository

```bash
git clone https://github.com/revanth2612/crop-disease-ai.git
cd crop-disease-ai
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run Backend (FastAPI)

```bash
uvicorn app:app --reload
```

API will be available at:

```
http://localhost:8000/docs
```

---

### 4. Run Frontend (Streamlit)

```bash
streamlit run ui.py
```

---

## 🐳 Docker Setup

### Build image

```bash
docker build -t crop-disease-ai .
```

### Run container

```bash
docker run -p 8000:8000 crop-disease-ai
```

---

## ☁️ Deployment

### Backend (Render)

* Deploy using Docker
* Exposes `/predict` API endpoint

### Frontend (Streamlit Cloud)

* Connect GitHub repo
* Update API URL in `ui.py`

```python
API_URL = "https://your-render-url.onrender.com/predict"
```

---

## 🔌 API Usage

### Endpoint

```
POST /predict
```

### Request

* `multipart/form-data`
* field: `file`

### Example Response

```json
{
  "detections": [
    {
      "class": "Pepper__bell___Bacterial_spot",
      "confidence": 0.87,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "advisory": "Remove infected leaves and apply fungicide."
}
```

---

## ⚠️ Limitations

* Model accuracy depends on training data
* Low-confidence predictions may occur
* LLM advisory is disabled in cloud (fallback used)

---

## 🔮 Future Work

* Improve model accuracy with larger datasets
* Integrate cloud-based LLM APIs (OpenAI/Gemini)
* Mobile app integration
* Multi-crop disease support
* Real-time video detection

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Revanth**

* GitHub: https://github.com/revanth2612

---

## ⭐ Support

If you find this project useful:

* ⭐ Star the repo
* 🍴 Fork it
* 📢 Share it

---

## 📌 Citation

If you use this project in research:

```
Crop Disease AI: A Real-Time Deep Learning and LLM-Integrated System for Plant Disease Detection and Advisory
```

---
