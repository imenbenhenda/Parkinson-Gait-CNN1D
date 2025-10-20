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
* **Preprocessing:** Signals were normalized, unified to a fixed length (`1000` points) by truncation or padding, and the training set was balanced using oversampling and noise augmentation.

---

## 📈 Model Optimization Journey
This project followed an iterative process to transform an initial unstable model into a robust and reliable solution.

* **Initial Challenge:** The first version of the model suffered from high instability and overfitting, making its predictions unreliable.
* **Improvement Strategy:** A series of optimizations were implemented:
    * **Methodology Correction:** Ensured correct data splitting **before** applying oversampling and augmentation to prevent data leakage, providing a reliable evaluation.
    * **Hyperparameter Tuning for Stability:** Adjusted learning rate (`0.0001`), batch size (`32`), and dropout rates (`0.3`) to achieve stable and smooth convergence during training.
    * **Data Handling:** Used `RandomOverSampler` for class balancing and added noise augmentation **only** on the training set. Keras callbacks (`ReduceLROnPlateau`, `EarlyStopping`, `ModelCheckpoint`) were used to manage training effectively and save the best performing model based on validation accuracy.
* **Result:** A final stable model that demonstrates good generalization (minimal overfitting) and achieves reliable performance. Random seeds were fixed for reproducibility.

---

## 📊 Final Performance
The final optimized model achieves the following performance on the **validation set** (using the weights saved at the best epoch by `ModelCheckpoint`):

| Metric                  | Value   |
| :---------------------- | :------ |
| **Accuracy (Validation)** | **87%** |
| **F1-Score (average)** | **0.88**|
| **Precision (Parkinson's)** | **91%** |
| **Recall (Parkinson's)** | **95%** |

*Note: The performance metrics are based on the model's evaluation on the validation split.*

---

## 🛠️ Model Architecture
The model is a sequential 1D Convolutional Neural Network (CNN) built with Keras, designed for time-series classification.

* **Input Layer:** Expects input sequences of shape `(1000, 16)`.
* **Two Convolutional Blocks:**
    1.  `Conv1D` (16 filters, kernel size 5, ReLU) -> `BatchNormalization` -> `MaxPooling1D`
    2.  `Conv1D` (32 filters, kernel size 3, ReLU) -> `BatchNormalization` -> `MaxPooling1D` -> `Dropout(0.3)`
* **Classification Head:** `GlobalAveragePooling1D` -> `Dense` (32 units, ReLU) -> `Dropout(0.3)` -> `Dense` (2 units, Softmax).
* **Optimizer & Loss Function:**
    * **Optimizer:** `Adam` (learning rate optimized to `0.0001`).
    * **Loss Function:** `categorical_crossentropy`.
* **Robust Training Strategy:** Training is managed using `EarlyStopping` (patience=15 on `val_accuracy`), `ModelCheckpoint` (saving the best model based on `val_accuracy`), and `ReduceLROnPlateau` (reducing learning rate on `val_loss` plateau).

---

## 🚀 Technologies Used
* Python
* TensorFlow / Keras
* Scikit-learn
* NumPy / Pandas
* Matplotlib / Seaborn
* Imbalanced-learn (`imblearn`)

---

## ⚙️ Installation & Usage
```bash
# 1. Clone the repository
git clone [https://github.com/imenbenhenda/Parkinson-Gait-CNN1D.git](https://github.com/imenbenhenda/Parkinson-Gait-CNN1D.git)
cd Parkinson-Gait-CNN1D

# 2. Create and activate a virtual environment (Recommended)
# python -m venv venv
# .\venv\Scripts\activate  # On Windows PowerShell
# source venv/bin/activate # On macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run training (Generates models/best_model.keras and results/training_curves.png)
python src/train_model.py

