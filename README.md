# Parkinson's Disease Detection from Gait Analysis

## Objective
Develop a Deep Learning (1D-CNN) model capable of detecting Parkinson's disease from ground reaction force (VGRF) signals measured during walking.

## Dataset
- Source: [PhysioNet GaitPDB](https://physionet.org/content/gaitpdb/1.0.0/)
- VGRF walking sensors (16 sensors per foot)
- Time-series data from healthy individuals and Parkinson's patients
- Preprocessing: signal normalization and padding

## Results
| Metric                 | Value |
|------------------------|-------|
| Accuracy               | 87%   |
| F1-Score (average)     | 0.87  |
| Precision (Parkinson's)| 86%   |
| Recall (Parkinson's)   | 88%   |

## Model Architecture
- 1D-CNN network with:
  - 2 convolutional layers + max pooling
  - Dense layer + Dropout
  - Softmax output for 2 classes (Parkinson's / Healthy)
- Optimizer: Adam
- Loss: categorical_crossentropy
- Callbacks: EarlyStopping and ModelCheckpoint

## Key Features
- 🧠 **Neurological Analysis** - Parkinson's detection from gait patterns
- 📈 **Time Series Processing** - 1D-CNN for sensor signal analysis
- 🎯 **High Accuracy** - 87% detection performance
- 🔬 **Medical Research** - Ground reaction force signal processing

## Technologies Used
- Python
- TensorFlow/Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pandas

## Installation & Usage
```bash
# Clone repository
git clone https://github.com/your-username/parkinson-gait-detection.git
cd parkinson-gait-detection

# Install dependencies
pip install -r requirements.txt

# Run training
python scripts/train_model.py

# Test predictions
python scripts/test_model.py
Project Structure
parkinson-gait-detection/
├── scripts/
│   ├── train_model.py    # Model training script
│   └── test_model.py     # Inference and testing
├── models/               # Trained models
├── data/                 # Dataset and processed signals
├── notebooks/            # Data exploration and analysis
└── results/              # Performance metrics and graphs
Reproduction Instructions:
Place the data in the data/ folder
Launch training: python scripts/train_model.py
View generated results and graphs
Test predictions on examples: python scripts/test_model.py

Author
Imen Ben Henda - Computer Engineering Student
