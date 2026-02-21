import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Eye Detector")

st.title("🎯 Eye Detector")

TARGET_RADIUS = 60
MOTION_THRESHOLD = 5000


class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=25
        )

    def draw_target(self, frame):
        h, w, _ = frame.shape
        center = (w // 2, h // 2)

        cv2.circle(frame, center, TARGET_RADIUS, (0, 0, 255), 2)

        return center

    def detect_motion(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)

        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        motion_pixels = np.sum(thresh == 255)

        return thresh, motion_pixels

    def motion_in_target(self, mask, center):
        h, w = mask.shape
        y, x = np.ogrid[:h, :w]

        circle = (x - center[0])**2 + (y - center[1])**2 <= TARGET_RADIUS**2

        return np.any(mask[circle] == 255)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        motion_mask, motion_pixels = self.detect_motion(img)

        center = self.draw_target(img)

        if motion_pixels > MOTION_THRESHOLD:
            if self.motion_in_target(motion_mask, center):
                cv2.putText(
                    img,
                    "PERDU",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    3
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="eye-detector",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
