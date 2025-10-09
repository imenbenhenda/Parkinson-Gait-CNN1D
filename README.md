# Parkinson's Disease Detection from Gait Analysis

## Objective
Develop a Deep Learning (1D-CNN) model capable of detecting Parkinson's disease from ground reaction force (VGRF) signals measured during walking.

## Dataset
- Source: [PhysioNet GaitPDB](https://physionet.org/content/gaitpdb/1.0.0/)
- VGRF walking sensors (16 sensors per foot)
- Time-series data from healthy individuals and Parkinson's patients
- Preprocessing: signal normalization and padding

📈 ** Model Optimization Journey ** 
This project followed an iterative process to transform an initial unstable model into a robust and reliable solution.

Initial Challenge: The first version of the model suffered from high instability and overfitting, making its predictions unreliable.

Improvement Strategy: A series of optimizations were implemented:

Enhanced Regularization: Use of multiple Dropout layers with adjusted rates (0.4 and 0.5) to prevent overfitting.

Training Refinement: Optimization of the learning rate (set to 0.0003) and use of Keras callbacks (ReduceLROnPlateau, EarlyStopping) to stabilize and optimize convergence.

Data Enhancement: Application of oversampling with RandomOverSampler to balance the classes and use of data augmentation (adding noise) to create a more robust dataset.

Result: A final stable model that does not overfit and whose performance is reliable and reproducible.
Author
Imen Ben Henda - Computer Engineering Student
