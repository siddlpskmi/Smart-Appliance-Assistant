import streamlit as st
import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
from qa_engine import ApplianceQASystem


# ----------------------------
# Global TTS engine
# ----------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 175)
engine.setProperty("volume", 1.0)


def speak(text: str):
    if not text:
        return
    engine.say(text)
    engine.runAndWait()


# ----------------------------
# Load YOLO model and QA system once
# ----------------------------
@st.cache_resource
def load_yolo(model_path: str = "best.pt"):
    return YOLO(model_path)


@st.cache_resource
def load_qa(kb_path: str = "kb.csv"):
    return ApplianceQASystem(kb_path)


model = load_yolo("best.pt")
qa_system = load_qa("kb.csv")


# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="UniQ: Smart Appliance Assistant", layout="centered")
st.title("🤖 UniQ – Smart Vision-Based Appliance Assistant")

st.write(
    "Use your laptop camera to detect an appliance, lock it, and then ask UniQ "
    "questions about how to use, clean, or safely operate it."
)

# ----------------------------
# Session state
# ----------------------------
if "last_detected" not in st.session_state:
    st.session_state.last_detected = None

if "last_conf" not in st.session_state:
    st.session_state.last_conf = None

if "locked_device" not in st.session_state:
    st.session_state.locked_device = None


# ----------------------------
# 1️⃣ Device detection with camera preview
# ----------------------------
st.header("1️⃣ Device Detection")

st.write(
    "1. Hold **one device** (Mouse, KeyboardLayer, Food_Processor, Water_Heater) in front of the camera.\n"
    "2. Use the camera widget to preview and capture a frame.\n"
    "3. UniQ will run YOLO on the captured frame and show the detection.\n"
    "4. If it looks correct, click **Lock this device for Q&A**."
)

# Streamlit camera widget: shows live preview + lets you capture a frame
camera_image = st.camera_input("📷 Live Camera Preview (take a snapshot when ready)")

CONF_THRESHOLD = 0.75  # require reasonably high confidence

if camera_image is not None:
    # Convert captured image to OpenCV format (BGR)
    bytes_data = camera_image.getvalue()
    np_img = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Run YOLO
    results = model(frame, conf=CONF_THRESHOLD)
    annotated = results[0].plot()

    # Show annotated image in Streamlit
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Detection result", use_container_width=True)

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        confs = boxes.conf.cpu().numpy()
        best_idx = confs.argmax()
        best_conf = float(confs[best_idx])
        cls_id = int(boxes.cls[best_idx].item())
        label = model.names[cls_id]

        st.session_state.last_detected = label
        st.session_state.last_conf = best_conf

        if best_conf >= CONF_THRESHOLD and label != "undefined":
            st.success(f"Detected device: **{label}** (confidence {best_conf:.2f})")
        else:
            st.session_state.last_detected = None
            st.warning(
                f"Detected something ({label}, {best_conf:.2f}) but not confidently enough "
                "or it is 'undefined'. Try capturing again with the device closer, centered, "
                "and better lit."
            )
    else:
        st.session_state.last_detected = None
        st.warning("No device detected in this frame. Try again with the device closer or better lit.")

# Lock button
if st.session_state.last_detected:
    st.write(
        f"Last detected device: **{st.session_state.last_detected}** "
        f"(confidence {st.session_state.last_conf:.2f})"
    )
    if st.button("🔒 Lock this device for Q&A"):
        if st.session_state.last_detected != "undefined":
            st.session_state.locked_device = st.session_state.last_detected
            st.success(f"Locked device: **{st.session_state.locked_device}**")
        else:
            st.warning("Detected class is 'undefined'. Show a known device and capture again.")
else:
    st.info("Capture a frame first so UniQ can detect a device.")


# ----------------------------
# 2️⃣ Question Answering
# ----------------------------
st.write("---")
st.header("2️⃣ Question Answering")

if st.session_state.locked_device:
    st.success(f"Current locked device: **{st.session_state.locked_device}**")
else:
    st.warning("No device locked yet. Please detect and lock a device first.")

st.write(
    "Ask a question about the locked device. You can start with a wake phrase like:\n"
    "- `hey uniq, how do I turn this on?`\n"
    "- `how to clean this?`\n"
    "- `is this safe to use?`"
)

question = st.text_input("💬 Your question")

col1, col2 = st.columns(2)
with col1:
    ask_clicked = st.button("🤖 Ask UniQ")
with col2:
    clear_clicked = st.button("♻️ Clear locked device")

if clear_clicked:
    st.session_state.locked_device = None
    st.info("Locked device cleared. Detect and lock again to continue.")

if ask_clicked:
    if not st.session_state.locked_device:
        st.error("Please lock a device first before asking a question.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        text = question.strip()
        lower = text.lower()
        wake = "hey uniq"
        if lower.startswith(wake):
            text = text[len(wake):].strip()

        device = st.session_state.locked_device
        answer = qa_system.answer(device, text)

        st.subheader("🧠 UniQ's Answer")
        st.write(answer)
        speak(answer)
