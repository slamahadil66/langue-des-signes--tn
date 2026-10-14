import os
import sys
import cv2
import numpy as np
import mediapipe as mp

SIGNES = [
    "asslema", "labess", "ca_va", "hamdoulleh", "netcharfou", "alech",
    "je_jure", "attend", "fierte", "je_te_punis", "nhebek", "eni",
    "oui", "merci", "yemchi", "nekl", "akther", "toilette",
]
NB_SEQUENCES = 30   # séquences par signe
LONGUEUR = 30       # images par séquence
DOSSIER = "data"

# Haut du corps : nez, épaules, coudes, poignets
CORPS = [0, 11, 12, 13, 14, 15, 16]
# Visage : bouche, sourcils, yeux, nez, menton, côtés
VISAGE = [61, 291, 0, 17, 13, 14, 78, 308,
          70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
          33, 133, 159, 145, 362, 263, 386, 374,
          1, 4, 152, 234, 454]

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils


def points(landmarks, indices=None):
    """Convertit des landmarks en vecteur plat (x, y, z). Des zéros si absents."""
    nb = len(indices) if indices is not None else 21
    if landmarks is None:
        return np.zeros(nb * 3)
    pts = landmarks.landmark
    if indices is not None:
        pts = [pts[i] for i in indices]
    return np.array([[p.x, p.y, p.z] for p in pts]).flatten()


def extraire(results):
    """Une image -> 240 nombres : corps (21) + visage (93) + main G (63) + main D (63)."""
    return np.concatenate([
        points(results.pose_landmarks, CORPS),
        points(results.face_landmarks, VISAGE),
        points(results.left_hand_landmarks),
        points(results.right_hand_landmarks),
    ])


def lire(cap, holistic):
    ok, frame = cap.read()
    if not ok:
        return None, None
    frame = cv2.flip(frame, 1)
    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_draw.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return frame, results


def afficher(frame, texte, couleur=(0, 200, 0)):
    cv2.putText(frame, texte, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)
    cv2.imshow("Collecte Holistic", frame)
    return cv2.waitKey(1) & 0xFF


def enregistrer(signe, cap, holistic):
    """Enregistre les séquences manquantes d'un signe. False = quitter."""
    chemin = os.path.join(DOSSIER, signe)
    os.makedirs(chemin, exist_ok=True)
    n = len(os.listdir(chemin))   # reprend après les fichiers existants

    while n < NB_SEQUENCES:
        # 1) Attente : tu te places, puis ESPACE
        while True:
            frame, _ = lire(cap, holistic)
            if frame is None:
                return False
            touche = afficher(frame, f"{signe} {n}/{NB_SEQUENCES} | ESPACE=demarrer s=passer q=quitter")
            if touche == ord(" "):
                break
            if touche == ord("s"):
                return True
            if touche == ord("q"):
                return False

        # 2) Préparation (environ 1 seconde)
        for _ in range(20):
            frame, _ = lire(cap, holistic)
            if frame is None:
                return False
            afficher(frame, "Prepare-toi...", (0, 165, 255))

        # 3) Enregistrement : LONGUEUR images
        sequence = []
        images_avec_main = 0
        for i in range(LONGUEUR):
            frame, results = lire(cap, holistic)
            if frame is None:
                return False
            sequence.append(extraire(results))
            if results.left_hand_landmarks or results.right_hand_landmarks:
                images_avec_main += 1
            afficher(frame, f"ENREGISTREMENT {i + 1}/{LONGUEUR}", (0, 0, 255))

        # 4) Contrôle qualité : au moins la moitié des images avec une main
        if images_avec_main < LONGUEUR // 2:
            print(f"Rejete : main vue sur {images_avec_main}/{LONGUEUR} images, refais ce geste.")
            continue

        np.save(os.path.join(chemin, f"{n}.npy"), np.array(sequence))
        print(f"Sauvegarde : {signe}/{n}.npy")
        n += 1
    return True


def main():
    demandes = sys.argv[1:]
    if not demandes:
        print("Usage : python collect_holistic.py signe1 signe2 ...")
        print("Signes disponibles :", ", ".join(SIGNES))
        return
    inconnus = [s for s in demandes if s not in SIGNES]
    if inconnus:
        print("Signes inconnus :", ", ".join(inconnus))
        return

    cap = cv2.VideoCapture(0)
    # model_complexity=0 si ton PC est lent
    with mp_holistic.Holistic(model_complexity=1,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        for signe in demandes:
            if not enregistrer(signe, cap, holistic):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()