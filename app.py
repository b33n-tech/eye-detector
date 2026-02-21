# app.py
import streamlit as st
import cv2
import numpy as np

st.title("Détection de cible - Prototype")

# Activation caméra
run = st.checkbox('Activer la caméra')

FRAME_WINDOW = st.image([])  # Placeholder pour vidéo

# Définir la zone cible (cercle au centre)
def draw_target(frame):
    h, w, _ = frame.shape
    center = (w // 2, h // 2)
    radius = 50
    color = (0, 0, 255)  # rouge
    thickness = 2
    cv2.circle(frame, center, radius, color, thickness)
    return center, radius

# Vérifier si un mouvement touche la cible
def is_touching_target(motion_mask, center, radius):
    h, w = motion_mask.shape
    y, x = np.ogrid[:h, :w]
    mask_circle = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    return np.any(motion_mask[mask_circle] > 0)

# Détection simple via soustraction de fond
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25)

if run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    perdu_text = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Impossible d'accéder à la caméra")
            break

        frame = cv2.flip(frame, 1)  # miroir pour smartphone
        motion_mask = bg_subtractor.apply(frame)
        motion_mask = cv2.threshold(motion_mask, 127, 255, cv2.THRESH_BINARY)[1]

        # Dessiner cible
        center, radius = draw_target(frame)

        # Vérifier mouvement sur cible
        if is_touching_target(motion_mask, center, radius):
            perdu_text.markdown("<h1 style='color:red;'>PERDU !</h1>", unsafe_allow_html=True)
        else:
            perdu_text.empty()

        # Afficher vidéo
        FRAME_WINDOW.image(frame, channels="BGR")
