import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

st.markdown(
    """
    <style>
    /* Background styling */
    body {
        background-image: url('https://images.unsplash.com/photo-1525159831892-d7cb0fbb3a0c?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=1080'); /* Replace with your desired URL */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }

    /* Header styling */
    h1, h2, h3 {
        font-family: 'Trebuchet MS', sans-serif;
        color: #FFAA33;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    }

    /* Input box styling */
    input {
        border: 2px solid #FFAA33;
        border-radius: 10px;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }

    /* Button styling */
    .stButton>button {
        background-color: #FFAA33;
        color: white;
        font-size: 18px;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.4);
    }

    .stButton>button:hover {
        background-color: #FF7722;
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model = load_model("model.h5")
label = np.load("labels.npy")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("🎵Emotion-Based Music Recommender🎵")

if "run" not in st.session_state:
    st.session_state["run"] = True

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = True
else:
    st.session_state["run"] = False

lang = st.text_input("🎤 Enter your preferred language:")
singer = st.text_input("🎼 Enter your favorite singer:")

start_camera = st.button("📸 Start Camera")
stop_camera = st.button("🛑 Stop Camera")

if start_camera:
    st.session_state["run"] = True
    st.warning("Please allow access to your camera.")
    cap = cv2.VideoCapture(0)

    st_frame = st.empty()
    while st.session_state["run"]:
        ret, frm = cap.read()
        if not ret:
            st.error("Failed to access the camera.")
            break

        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(
            frm,
            res.face_landmarks,
            holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
            connection_drawing_spec=drawing.DrawingSpec(thickness=1),
        )
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        st_frame.image(frm, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

if stop_camera:
    st.session_state["run"] = False

btn = st.button("🎶 Recommend me songs!")
if btn:
    if not emotion:
        st.warning("Please let me capture your emotion first.")
        st.session_state["run"] = True
    else:
        webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = False
