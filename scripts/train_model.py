import os
import numpy as np
import matplotlib.pyplot as plt
import collections
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# Pour oversampling
from imblearn.over_sampling import RandomOverSampler

# --- Import des fonctions de prétraitement ---
from load_data import load_demographics, get_file_labels
from preprocess_signals import read_signals, normalize_signals, pad_signals

# --- 1. Charger les données ---
project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_path, "../data")

demographics = load_demographics(data_path)
all_files, file_labels = get_file_labels(data_path, demographics)

X, y = read_signals(all_files, file_labels)
X_norm = normalize_signals(X)
X_array = pad_signals(X_norm)
y_array = np.array(y)

print("Répartition des classes:", collections.Counter(y_array))

# --- Oversampling pour équilibrer ---
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X_array.reshape(len(X_array), -1), y_array)
X_resampled = X_resampled.reshape(-1, X_array.shape[1], X_array.shape[2])

print("Après oversampling:", collections.Counter(y_resampled))

# --- One-hot encoding ---
y_cat = to_categorical(y_resampled, num_classes=2)

# --- Split stratifié (20% validation) ---
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_cat,
    test_size=0.2,
    stratify=y_resampled,
    random_state=42
)

# --- Class weights (moins utile car oversampling, mais je laisse) ---
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_resampled),
    y=y_resampled
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# --- 2. Définir un modèle CNN plus léger (moins de filtres + régularisation) ---
model = Sequential([
    Input(shape=(X_array.shape[1], X_array.shape[2])),

    Conv1D(16, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),

    Conv1D(32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),

    GlobalAveragePooling1D(),

    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 3. Callbacks ---
checkpoint_path = os.path.join(project_path, "../models/best_model.keras")
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# --- 4. Entraînement ---
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, reduce_lr, early_stop],
    class_weight=class_weights
)

# --- Sauvegarde finale du modèle ---
model.save(os.path.join(project_path, "../models/parkinson_cnn.keras"))

# --- 5. Évaluation ---
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

print("\n Rapport de classification :")
print(classification_report(y_true, y_pred_classes, target_names=["Contrôle", "Parkinson"]))

# --- 6. Tracé des courbes ---
plt.figure(figsize=(10,5))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Train Loss', linestyle='--')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')

plt.xlabel("Epochs")
plt.ylabel("Valeur")
plt.title("Courbes Accuracy & Loss (CNN Parkinson)")
plt.legend()
plt.grid(True)
plt.show()
