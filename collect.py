import os
import sys
import time

import cv2
import numpy as np
import mediapipe as mp

SEQ_LEN = 30          # nombre d'images par séquence
COUNTDOWN = 2         # secondes avant l'enregistrement
DATA_DIR = "data"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def extract_keypoints(results):
    """Renvoie un vecteur de 126 nombres : main gauche (63) + main droite (63).
    Si une main n'est pas détectée, ses 63 valeurs restent à 0."""
    left = np.zeros(63)
    right = np.zeros(63)
    if results.multi_hand_landmarks:
        for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
            if handed.classification[0].label == "Left":
                left = coords
            else:
                right = coords
    return np.concatenate([left, right])


def main():
    if len(sys.argv) < 2:
        print("Utilisation : python collect.py <nom_du_signe>")
        return

    sign = sys.argv[1]
    folder = os.path.join(DATA_DIR, sign)
    os.makedirs(folder, exist_ok=True)
    count = len(os.listdir(folder))   # numéro de la prochaine séquence

    cap = cv2.VideoCapture(0)
    state = "idle"                    # idle -> countdown -> recording
    frames = []
    start_time = 0

    with mp_hands.Hands(max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            if state == "countdown":
                remaining = COUNTDOWN - (time.time() - start_time)
                if remaining > 0:
                    msg = f"Preparez-vous : {int(remaining) + 1}"
                else:
                    state = "recording"
                    frames = []
                    msg = "GO"
            elif state == "recording":
                frames.append(extract_keypoints(results))
                msg = f"Enregistrement {len(frames)}/{SEQ_LEN}"
                if len(frames) == SEQ_LEN:
                    np.save(os.path.join(folder, f"{count}.npy"), np.array(frames))
                    print(f"Sauvegarde : {sign}/{count}.npy")
                    count += 1
                    state = "idle"
            else:
                msg = "s = enregistrer | q = quitter"

            cv2.putText(frame, f"Signe : {sign} ({count} sequences)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, msg, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.imshow("Collecte", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") and state == "idle":
                state = "countdown"
                start_time = time.time()
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()