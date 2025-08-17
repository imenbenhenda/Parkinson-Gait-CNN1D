# Détection de la Maladie de Parkinson à partir de la marche

## Objectif
Développer un modèle de Deep Learning (CNN 1D) capable de détecter la maladie de Parkinson à partir de signaux de marche mesurés par capteurs de force au sol (VGRF).

## Dataset
- Capteurs de marche VGRF (16 capteurs par pied)
- Données temporelles de personnes saines et atteintes de Parkinson
- Prétraitement : normalisation et padding des signaux

##  Résultats
| Métrique       | Valeur |
|---------------|--------|
| Accuracy      | 89.5%  |
| F1-Score (moy)| 0.80   |
| Précision (Parkinson) | 91% |
| Rappel (Parkinson)    | 67% |

##  Architecture du modèle

_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 Conv1D (16 filters)         (None, 2636, 16)          784       
 BatchNormalization          (None, 2636, 16)          64        
 MaxPooling1D                (None, 1318, 16)          0         
 Dropout (0.5)               (None, 1318, 16)          0         
 Conv1D (32 filters)         (None, 1316, 32)          1568      
 GlobalAveragePooling1D      (None, 32)                0         
 Dense (32 units)            (None, 32)                1056      
 Dense (2 units)             (None, 2)                 66        
=================================================================

## Modèle
- Réseau 1D-CNN avec :
  - 2 couches convolutionnelles + max pooling
  - Dense layer avec régularisation L2 et Dropout
  - Sortie softmax pour 2 classes (Parkinson / sain)
- Optimiseur : Adam
- Loss : categorical_crossentropy
- Callback : EarlyStopping et ModelCheckpoint

## Instructions pour reproduire
1. Installer les dépendances : `pip install -r requirements.txt`
2. Placer les données dans le dossier `data/`
3. Lancer l'entraînement : `python scripts/train_model.py`
4. Voir les résultats et graphiques générés
5. Tester des prédictions sur quelques exemples : `python scripts/test_model.py`
