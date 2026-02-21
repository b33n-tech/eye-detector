import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Eye Detector", layout="centered")

st.title("🎯 Eye Detector")

st.write("Bouge un objet dans la cible pour perdre.")

TARGET_RADIUS = 70
MOTION_THRESHOLD = 800  # Ajustable selon sensibilité


class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=25
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        h, w, _ = img.shape
        center = (w // 2, h // 2)

        # -------------------------
        # MASQUE CIBLE
        # -------------------------
        target_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(target_mask, center, TARGET_RADIUS, 255, -1)

        # -------------------------
        # DETECTION MOUVEMENT
        # -------------------------
        fg_mask = self.bg_subtractor.apply(img)

        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # -------------------------
        # LIMITATION A LA CIBLE
        # -------------------------
        target_motion = cv2.bitwise_and(thresh, thresh, mask=target_mask)

        motion_pixels = np.sum(target_motion == 255)

        # -------------------------
        # DESSIN CIBLE
        # -------------------------
        cv2.circle(img, center, TARGET_RADIUS, (0, 0, 255), 2)

        # -------------------------
        # LOGIQUE PERDU
        # -------------------------
        if motion_pixels > MOTION_THRESHOLD:
            cv2.putText(
                img,
                "PERDU",
                (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                4
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="eye-detector",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)
