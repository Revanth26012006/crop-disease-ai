import streamlit as st
import requests
from PIL import Image, ImageDraw

st.set_page_config(page_title="Crop AI", layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
.block-container { padding-top: 2rem; }

.title {
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: #2e7d32;
}

.section {
    background-color: #111;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
}

.advisory-box {
    height: 240px;
    overflow-y: auto;
    background-color: #0e1117;
    padding: 10px;
    border-radius: 8px;
}

.chat-box {
    height: 320px;
    overflow-y: auto;
    background-color: #0e1117;
    padding: 10px;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
}

.chat-user {
    background-color: #2e7d32;
    padding: 8px;
    border-radius: 8px;
    margin: 5px;
    color: white;
    align-self: flex-end;
}

.chat-ai {
    background-color: #333;
    padding: 8px;
    border-radius: 8px;
    margin: 5px;
    color: white;
    align-self: flex-start;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown('<div class="title">🌿 Crop Disease AI Dashboard</div>', unsafe_allow_html=True)

# ---------- STATE ----------
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- LAYOUT ----------
left, right = st.columns([1, 2])

# ================= LEFT =================
with left:
    st.markdown('<div class="section">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

    analyze = st.button("Analyze", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ================= RIGHT =================
with right:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("🧪 AI Analysis")

    if analyze and uploaded_file:
        with st.spinner("Analyzing..."):
            res = requests.post(
                "https://crop-disease-ai-7mwe.onrender.com/predict",
                files={"file": uploaded_file.getvalue()}
            )
            st.session_state.result = res.json()

    if st.session_state.result:
        data = st.session_state.result

        if data.get("detections"):
            det = data["detections"][0]

            disease = det["class"]
            confidence = det["confidence"]
            bbox = det["bbox"]

            # 🔥 FIX: Improve display label
            if confidence < 0.2:
                display_disease = f"⚠️ Low confidence prediction"
            else:
                display_disease = disease

            col1, col2 = st.columns([1.5, 1])

            # IMAGE
            with col1:
                img = Image.open(uploaded_file)
                draw = ImageDraw.Draw(img)

                x1, y1, x2, y2 = bbox

                # FIX BBOX
                x_min = int(min(x1, x2))
                y_min = int(min(y1, y2))
                x_max = int(max(x1, x2))
                y_max = int(max(y1, y2))

                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
                st.image(img, use_container_width=True)

            # DETAILS
            with col2:
                st.markdown("### 🌱 Disease")
                st.write(display_disease)

                st.markdown("### 📊 Confidence")
                st.write(f"{confidence:.2f}")

        # ---------- ADVISORY ----------
        if data.get("advisory"):
            st.markdown("### 💡 Advisory")
            st.markdown(
                f'<div class="advisory-box">{data["advisory"]}</div>',
                unsafe_allow_html=True
            )

    else:
        st.info("Upload image and click Analyze")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= CHAT =================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("💬 Ask AI")

user_input = st.text_input("Ask about the disease...")

if st.button("Send"):
    if user_input:
        st.session_state.chat_history.append(("user", user_input))

        disease_context = ""
        if st.session_state.result and st.session_state.result.get("detections"):
            disease_context = st.session_state.result["detections"][0]["class"]

        prompt = f"""
You are an agriculture expert.

Disease: {disease_context}

User question: {user_input}
"""

        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "phi", "prompt": prompt, "stream": False}
        )

        reply = res.json()["response"]
        st.session_state.chat_history.append(("ai", reply))

# CHAT DISPLAY
chat_html = '<div class="chat-box">'
for role, msg in st.session_state.chat_history:
    if role == "user":
        chat_html += f'<div class="chat-user">🧑 {msg}</div>'
    else:
        chat_html += f'<div class="chat-ai">🤖 {msg}</div>'
chat_html += '</div>'

st.markdown(chat_html, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)