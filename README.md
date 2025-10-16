# Parkinson's Disease Detection from Gait Analysis

## 🎯 Objective
To develop a Deep Learning (1D-CNN) model capable of detecting Parkinson's disease from Vertical Ground Reaction Force (VGRF) signals measured during walking.

---

## 💾 Dataset
* **Source:** The data used in this project is from the Gait in Parkinson's Disease Database available on [PhysioNet](https://physionet.org/content/gaitpdb/1.0.0/).
* **License**: Open Data Commons Attribution License v1.0  
* **Ethical Approval**: Data anonymized and collected with patient consent  
* **Citations**: Goldberger et al. (2000) - Standard PhysioNet citation
* **Content:** It consists of time-series data from 16 underfoot sensors, measuring the Vertical Ground Reaction Force (VGRF) of both healthy individuals and patients with Parkinson's disease.
* **Preprocessing:** The signals were preprocessed through normalization, padding to a uniform length, and balancing using oversampling techniques.

---

## 📈 Model Optimization Journey
This project followed an iterative process to transform an initial unstable model into a robust and reliable solution.

* **Initial Challenge:** The first version of the model suffered from high instability and overfitting, making its predictions unreliable.
* **Improvement Strategy:** A series of optimizations were implemented:
    * **Enhanced Regularization:** Use of multiple `Dropout` layers with adjusted rates (0.4 and 0.5) to prevent overfitting.
    * **Training Refinement:** Optimization of the learning rate (set to `0.0003`) and use of Keras callbacks (`ReduceLROnPlateau`, `EarlyStopping`) to stabilize and optimize convergence.
    * **Data Enhancement:** Application of oversampling with `RandomOverSampler` to balance the classes and use of data augmentation (adding noise) to create a more robust dataset.
* **Result:** A final stable model that does not overfit and whose performance is reliable and reproducible.

---

## 📊 Final Performance
The optimized model achieves the following performance on the test set:

| Metric                 | Value   |
| :--------------------- | :------ |
| **Accuracy** | **87%** |
| **F1-Score (average)** | **0.87**|
| **Precision (Parkinson's)** | **84%** |
| **Recall (Parkinson's)** | **91%** |

---

## 🛠️ Model Architecture
The model is a sequential 1D Convolutional Neural Network (CNN) built with Keras.

* **Two Convolutional Blocks:** Each block consists of a `Conv1D` layer (ReLU), a `BatchNormalization` layer, and a `MaxPooling1D` layer.
* **Strong Regularization:** `Dropout` layers are integrated to prevent overfitting.
* **Classification Head:** A `GlobalAveragePooling1D` layer followed by `Dense` layers with a `Softmax` activation for the final classification.
* **Optimizer & Loss Function:**
    * **Optimizer:** `Adam` (learning rate of `0.0003`).
    * **Loss Function:** `categorical_crossentropy`.
* **Robust Training Strategy:** Use of `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau`.

---

## 🚀 Technologies Used
* Python
* TensorFlow / Keras
* Scikit-learn
* NumPy / Pandas
* Matplotlib / Seaborn

---

## ⚙️ Installation & Usage
```bash
# 1. Clone the repository
git clone (https://github.com/imenbenhenda/Parkinson-Gait-CNN1D.git)
cd parkinson-gait-detection

# 2. Run training 
python scripts/train_model.py

# 3. Evaluate the best saved model
python scripts/test_model.py
