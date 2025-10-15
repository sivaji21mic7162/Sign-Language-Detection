import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from gtts import gTTS
import tempfile
import streamlit as st
import os

# ------------------ Labels ------------------
sign_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'Apple', 'Can', 'Get', 'Good', 'Give me a call', 'I love you',
               'I want money', 'I want to go to the washroom', 'Please stop',
               'Thank you very much']

# ------------------ Load Models ------------------
STGCN_MODEL_PATH = "stgcn_model.h5"
YOLO_MODEL_PATH = "yolo11_best.pt"

@st.cache_resource
def load_models():
    stgcn_model = load_model(STGCN_MODEL_PATH)
    yolo_model = YOLO(YOLO_MODEL_PATH)
    return stgcn_model, yolo_model

stgcn_model, yolo_model = load_models()

# ------------------ Mediapipe Pose ------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ------------------ Text-to-Speech ------------------
def text_to_speech(text):
    """Converts text to speech and returns audio bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(tmp.name)
        return tmp.name

# ------------------ Extract Skeleton ------------------
def extract_skeleton(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.append([lm.x, lm.y, lm.z])
        return np.array(keypoints)
    return None

# ------------------ Predict Sign ------------------
def predict_sign(keypoints):
    x = np.expand_dims(keypoints, axis=0)  # batch
    x = np.expand_dims(x, axis=0)          # time dimension = 1
    x = np.expand_dims(x, axis=-1)         # channel
    prediction = stgcn_model.predict(x, verbose=0)
    label_id = np.argmax(prediction)
    return sign_labels[label_id]

# ------------------ YOLO Detection ------------------
CONFIDENCE_THRESHOLD = 0.50
GREEN = (0, 255, 0)

def detect_hand(frame):
    detections = yolo_model(frame)[0]
    hand_bbox = None
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        cls_id = data[5]
        if float(confidence) >= CONFIDENCE_THRESHOLD:
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            hand_bbox = (xmin, ymin, xmax, ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.putText(frame, f"{sign_labels[int(cls_id)]}", (xmin, ymin-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            break
    return frame, hand_bbox

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Sign Language Detection", page_icon="🤟", layout="wide")

# --- Custom CSS for Responsive Design ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #ffb347, #ff7f50);
        color: #333;
    }
    .title {
        text-align: center;
        font-size: 2rem;
        color: white;
        background-color: #ff7f50;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    @media (max-width: 768px) {
        .title {
            font-size: 1.4rem;
            padding: 8px;
        }
    }
    .stButton>button {
        width: 100%;
        background-color: #ff7f50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff9966;
        transform: scale(1.03);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🤟 Sign Language Detection (YOLO + ST-GCN)</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])
stframe = st.empty()
label_placeholder = st.empty()
audio_placeholder = st.empty()

if "run" not in st.session_state:
    st.session_state.run = False

with col1:
    st.write("### 🎮 Controls")
    if st.button("▶️ Start Detection"):
        st.session_state.run = True
    if st.button("⏹️ Stop Detection"):
        st.session_state.run = False

if st.session_state.run:
    cap = cv2.VideoCapture(0)
    last_label = ""

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ No camera found.")
            break

        frame, hand_bbox = detect_hand(frame)
        label = "Unknown"

        if hand_bbox:
            xmin, ymin, xmax, ymax = hand_bbox
            hand_crop = frame[ymin:ymax, xmin:xmax]
            keypoints = extract_skeleton(hand_crop)
            if keypoints is not None:
                try:
                    label = predict_sign(keypoints)
                except:
                    label = "unknown"

                if label != last_label and label != "unknown":
                    last_label = label
                    st.success(f"👐 Detected Sign: **{label}**")
                    # Convert to speech and play
                    audio_path = text_to_speech(label)
                    audio_placeholder.audio(audio_path, format="audio/mp3")

        stframe.image(frame, channels="BGR", use_container_width=True)

    cap.release()

