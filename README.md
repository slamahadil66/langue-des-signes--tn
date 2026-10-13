# Reconnaissance de la langue des signes tunisienne

Système de vision par ordinateur qui reconnaît en temps réel des signes de la
langue des signes tunisienne à partir d'une webcam et les traduit en texte.

**Statut : en cours de développement** (mini-projet de vision par ordinateur, ENET'Com).

## Objectif

Faciliter la communication avec les personnes sourdes ou malentendantes en
traduisant des gestes en texte, en temps réel et sans matériel spécialisé.

## Approche

Vidéo → points clés des mains → classification de séquence → texte affiché

1. **Capture** : lecture de la webcam avec OpenCV.
2. **Points clés** : extraction de 21 points par main (x, y, z) avec MediaPipe Hands.
3. **Dataset** : chaque signe est une séquence de 30 images, soit un tableau
   de forme `(30, 126)` enregistré en `.npy`.
4. **Prétraitement** : normalisation des coordonnées par rapport au poignet,
   augmentation des données.
5. **Modèles** : LSTM, puis Transformer, en PyTorch.
6. **Évaluation** : accuracy, F1, matrice de confusion, test sur des
   personnes absentes de l'entraînement.
7. **Démo temps réel** : affichage du signe prédit sur la vidéo avec OpenCV.

Utiliser les points clés plutôt que les images brutes rend le système plus
léger, plus rapide et insensible à l'arrière-plan.

## Technologies

Python 3.12, OpenCV, MediaPipe 0.10.14, NumPy, PyTorch, scikit-learn, Matplotlib.

## Structure du projet

langue-des-signes-tn/
├── test_webcam.py   # test de la détection des mains
├── collecte.py      # enregistrement des séquences de points clés
├── data/            # séquences .npy, un dossier par signe
└── requirements.txt


## Installation


python -m venv venv
venv\Scripts\activate
pip install mediapipe==0.10.14 "numpy<2" "opencv-python<4.11" "opencv-contrib-python<4.11"


## Utilisation

Tester la détection des mains :


python test_webcam.py


Collecter les données (ESPACE pour démarrer une séquence, `q` pour quitter) :

python collecte.py

## Feuille de route

- [x] Environnement et test MediaPipe
- [x] Script de collecte des points clés
- [ ] Collecte du dataset (8 à 15 signes, plusieurs personnes)
- [ ] Prétraitement et normalisation
- [ ] Entraînement LSTM puis Transformer
- [ ] Évaluation et analyse des erreurs
- [ ] Démo webcam en temps réel
- [ ] Extensions : phrases continues, visage et corps (MediaPipe Holistic)

## Auteure

Hadil, ENET'Com (Université de Sfax ), Systèmes Électroniques Intelligents.