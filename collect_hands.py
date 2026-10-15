import os
import sys
import cv2
import numpy as np
import mediapipe as mp

SIGNES = ["asslema", "toilette", "nekl", "attend",
          "oui", "merci", "akther", "netcharfou"]
NB_SEQUENCES = 30   # séquences par signe
LONGUEUR = 30       # images par séquence
DOSSIER = "data"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def extraire(results):
    """Une image -> 126 nombres (main gauche 63 + main droite 63)."""
    vecteur = np.zeros(126)
    if results.multi_hand_landmarks:
        for main, info in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[p.x, p.y, p.z] for p in main.landmark]).flatten()
            if info.classification[0].label == "Left":
                vecteur[:63] = coords
            else:
                vecteur[63:] = coords
    return vecteur


def lire(cap, hands):
    ok, frame = cap.read()
    if not ok:
        return None, None
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for main in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, main, mp_hands.HAND_CONNECTIONS)
    return frame, results


def afficher(frame, texte, couleur=(0, 200, 0)):
    cv2.putText(frame, texte, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)
    cv2.imshow("Collecte", frame)
    return cv2.waitKey(1) & 0xFF


def enregistrer(signe, cap, hands):
    """Enregistre les séquences manquantes d'un signe. False = quitter."""
    chemin = os.path.join(DOSSIER, signe)
    os.makedirs(chemin, exist_ok=True)
    n = len(os.listdir(chemin))   # reprend après les fichiers existants

    while n < NB_SEQUENCES:
        # 1) Attente : tu te places, puis ESPACE
        while True:
            frame, _ = lire(cap, hands)
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
            frame, _ = lire(cap, hands)
            if frame is None:
                return False
            afficher(frame, "Prepare-toi...", (0, 165, 255))

        # 3) Enregistrement : LONGUEUR images
        sequence = []
        images_avec_main = 0
        for i in range(LONGUEUR):
            frame, results = lire(cap, hands)
            if frame is None:
                return False
            sequence.append(extraire(results))
            if results.multi_hand_landmarks:
                images_avec_main += 1
            afficher(frame, f"ENREGISTREMENT {i + 1}/{LONGUEUR}", (0, 0, 255))

        # 4) Contrôle qualité : une main vue sur au moins la moitié des images
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
        print("Usage : python collect_hands.py signe1 signe2 ...")
        print("Signes disponibles :", ", ".join(SIGNES))
        return
    inconnus = [s for s in demandes if s not in SIGNES]
    if inconnus:
        print("Signes inconnus :", ", ".join(inconnus))
        return

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        for signe in demandes:
            if not enregistrer(signe, cap, hands):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()