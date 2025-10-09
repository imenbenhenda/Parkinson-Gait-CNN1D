Parkinson's Disease Detection from Gait Analysis
🎯 Objective
To develop a Deep Learning (1D-CNN) model capable of detecting Parkinson's disease from Vertical Ground Reaction Force (VGRF) signals measured during walking.

## Dataset
- Source: [PhysioNet GaitPDB](https://physionet.org/content/gaitpdb/1.0.0/)
- VGRF walking sensors (16 sensors per foot)
- Time-series data from healthy individuals and Parkinson's patients
- Preprocessing: signal normalization and padding

📈 Model Optimization Journey
This project followed an iterative process to transform an initial unstable model into a robust and reliable solution.
Initial Challenge: The first version of the model suffered from high instability and overfitting, making its predictions unreliable.
Improvement Strategy: A series of optimizations were implemented:
Enhanced Regularization: Use of multiple Dropout layers with adjusted rates (0.4 and 0.5) to prevent overfitting.
Training Refinement: Optimization of the learning rate (set to 0.0003) and use of Keras callbacks (ReduceLROnPlateau, EarlyStopping) to stabilize and optimize convergence.
Data Enhancement: Application of oversampling with RandomOverSampler to balance the classes and use of data augmentation (adding noise) to create a more robust dataset.
Result: A final stable model that does not overfit and whose performance is reliable and reproducible.
Before Optimization
![Uploading image.png…]()

After Optimization





Exporter vers Sheets

📊 Final Performance
The optimized model achieves the following performance on the test set:
Metric
Value
Accuracy
87%
F1-Score (average)
0.87
Precision (Parkinson's)
84%
Recall (Parkinson's)
91%

Exporter vers Sheets

🛠️ Model Architecture
The model is a sequential 1D Convolutional Neural Network (CNN) built with Keras.
Two Convolutional Blocks: Each block consists of a Conv1D layer (ReLU), a BatchNormalization layer, and a MaxPooling1D layer.
Strong Regularization: Dropout layers are integrated to prevent overfitting.
Classification Head: A GlobalAveragePooling1D layer followed by Dense layers with a Softmax activation for the final classification.
Optimizer & Loss Function:
Optimizer: Adam (learning rate of 0.0003).
Loss Function: categorical_crossentropy.
Robust Training Strategy: Use of EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau.

🚀 Technologies Used
Python
TensorFlow / Keras
Scikit-learn
NumPy / Pandas
Matplotlib / Seaborn

⚙️ Installation & Usage
Bash
# 1. Clone the repository
git clone https://github.com/your-username/parkinson-gait-detection.git
cd parkinson-gait-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training (to reproduce results)
python scripts/train_model.py

# 4. Evaluate the best saved model
python scripts/test_model.py


