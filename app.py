import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Eye Detector", layout="centered")

st.title("🎯 Eye Detector")

run = st.checkbox("Activer la caméra")

FRAME_WINDOW = st.image([])
status_text = st.empty()

# Paramètres
TARGET_RADIUS = 60
MOTION_THRESHOLD = 5000  # anti bruit / petits mouvements

def draw_target(frame):
    h, w, _ = frame.shape
    center = (w // 2, h // 2)

    cv2.circle(frame, center, TARGET_RADIUS, (0, 0, 255), 2)

    return center

def detect_motion(bg_subtractor, frame):
    fg_mask = bg_subtractor.apply(frame)

    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    motion_pixels = np.sum(thresh == 255)

    return thresh, motion_pixels

def motion_in_target(mask, center):
    h, w = mask.shape

    y, x = np.ogrid[:h, :w]

    circle = (x - center[0])**2 + (y - center[1])**2 <= TARGET_RADIUS**2

    return np.any(mask[circle] == 255)

if run:
    cap = cv2.VideoCapture(0)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=100,
        varThreshold=25
    )

    while run:
        ret, frame = cap.read()

        if not ret:
            st.warning("Impossible d'accéder à la caméra")
            break

        frame = cv2.flip(frame, 1)

        motion_mask, motion_pixels = detect_motion(bg_subtractor, frame)

        center = draw_target(frame)

        if motion_pixels > MOTION_THRESHOLD:
            if motion_in_target(motion_mask, center):
                status_text.markdown(
                    "<h1 style='color:red; text-align:center;'>PERDU</h1>",
                    unsafe_allow_html=True
                )
            else:
                status_text.empty()
        else:
            status_text.empty()

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()
