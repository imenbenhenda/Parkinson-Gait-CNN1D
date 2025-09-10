# Détection de la Maladie de Parkinson à partir de la marche

## Objectif
Développer un modèle de Deep Learning (CNN 1D) capable de détecter la maladie de Parkinson à partir de signaux de marche mesurés par capteurs de force au sol (VGRF).

## Dataset
- Source : [PhysioNet GaitPDB](https://physionet.org/content/gaitpdb/1.0.0/)
- Capteurs de marche VGRF (16 capteurs par pied)
- Données temporelles de personnes saines et atteintes de Parkinson
- Prétraitement : normalisation et padding des signaux

## Résultats
| Métrique                 | Valeur |
|---------------------------|--------|
| Accuracy                  | 87%    |
| F1-Score (moyenne)        | 0.87   |
| Précision (Parkinson)     | 86%    |
| Rappel (Parkinson)        | 88%    |

## Modèle
- Réseau 1D-CNN avec :
  - 2 couches convolutionnelles + max pooling
  - Dense layer + Dropout
  - Sortie softmax pour 2 classes (Parkinson / sain)
- Optimiseur : Adam
- Loss : categorical_crossentropy
- Callbacks : EarlyStopping et ModelCheckpoint
## Instructions pour reproduire
1. Placer les données dans le dossier `data/`
2. Lancer l'entraînement : `python scripts/train_model.py`
3. Voir les résultats et graphiques générés
4. Tester des prédictions sur quelques exemples : `python scripts/test_model.py`
