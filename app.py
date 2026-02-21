def recv(self, frame):
    img = frame.to_ndarray(format="bgr24")

    h, w, _ = img.shape
    center = (w // 2, h // 2)

    # ---- MASQUE CIBLE ----
    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.circle(mask, center, TARGET_RADIUS, 255, -1)

    # ---- DETECTION MOUVEMENT ----
    fg_mask = self.bg_subtractor.apply(img)

    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # ---- LIMITATION A LA CIBLE ----
    target_motion = cv2.bitwise_and(thresh, thresh, mask=mask)

    motion_pixels = np.sum(target_motion == 255)

    # ---- DESSIN CIBLE ----
    cv2.circle(img, center, TARGET_RADIUS, (0, 0, 255), 2)

    if motion_pixels > 800:  # seuil beaucoup plus bas maintenant
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
