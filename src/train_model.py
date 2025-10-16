import os
import numpy as np
import collections
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam # Importé ici pour une meilleure clarté

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

from imblearn.over_sampling import RandomOverSampler

# --- Import des fonctions externes (Assurez-vous qu'elles existent) ---
# NOTE: Le code suppose que 'load_data' et 'preprocess_signals' sont disponibles
from load_data import load_demographics, get_file_labels
from preprocess_signals import read_signals, normalize_signals, pad_signals, reduce_signal_length


# ====================================================================
#                     0. CONFIGURATION GLOBALE
# ====================================================================

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "../data")
MODELS_PATH = os.path.join(PROJECT_PATH, "../models")
RESULTS_PATH = os.path.join(PROJECT_PATH, "../results")

# Hyperparamètres de Préparation des Données
TARGET_SIGNAL_LENGTH = 1000
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Hyperparamètres du Modèle et de l'Entraînement
NUM_CLASSES = 2
LEARNING_RATE = 0.0003
EPOCHS = 40
BATCH_SIZE = 16
PATIENCE_EARLY_STOP = 20
PATIENCE_REDUCE_LR = 15
MIN_LR = 1e-7

# ====================================================================
#                     1. FONCTIONS DE PRÉTRAITEMENT & MODÈLE
# ====================================================================

def augment_time_series(signals, noise_factor=0.01):
    """
    Ajoute du bruit aux signaux pour augmenter la robustesse du modèle.
    """
    augmented = []
    for signal in signals:
        # Créer un bruit normal et l'ajouter au signal
        noise = np.random.normal(0, noise_factor, signal.shape)
        noisy_signal = signal + noise
        augmented.append(noisy_signal)
    return np.array(augmented)


def create_simplified_model(input_shape, num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE):
    """
    Définit et compile le modèle CNN 1D pour la classification de Parkinson.
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # Bloc 1
        Conv1D(16, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),
        
        # Bloc 2
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),
        
        # Classification
        GlobalAveragePooling1D(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def setup_callbacks():
    """
    Configure les Callbacks pour l'entraînement (Checkpoint, Early Stopping, Reduce LR).
    """
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
        
    checkpoint_path = os.path.join(MODELS_PATH, "best_model.keras")
    
    checkpoint = ModelCheckpoint(
        checkpoint_path, 
        monitor='val_accuracy', 
        save_best_only=True, 
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_accuracy', 
        patience=PATIENCE_EARLY_STOP, 
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=PATIENCE_REDUCE_LR,
        verbose=1,
        min_lr=MIN_LR
    )
    
    return [checkpoint, reduce_lr, early_stop]

# ====================================================================
#                     2. FONCTIONS DE TRAITEMENT PRINCIPAL
# ====================================================================

def load_and_preprocess_data():
    """
    Charge, normalise, réduit et augmente les données brutes.
    """
    print("--- 1. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES ---")
    
    # 1. Chargement initial et prétraitement
    demographics = load_demographics(DATA_PATH)
    all_files, file_labels = get_file_labels(DATA_PATH, demographics)
    X, y = read_signals(all_files, file_labels)
    X_norm = normalize_signals(X)
    X_array = pad_signals(X_norm)
    y_array = np.array(y)

    # Réduction de la taille
    print(f"Forme avant réduction: {X_array.shape}")
    X_array = reduce_signal_length(X_array, target_length=TARGET_SIGNAL_LENGTH)
    print(f"Forme après réduction: {X_array.shape}")
    print("Répartition des classes (initiale):", collections.Counter(y_array))

    # 2. Oversampling (rééquilibrage)
    ros = RandomOverSampler(random_state=RANDOM_SEED)
    X_resampled, y_resampled = ros.fit_resample(X_array.reshape(len(X_array), -1), y_array)
    X_resampled = X_resampled.reshape(-1, X_array.shape[1], X_array.shape[2])
    print("Après oversampling:", collections.Counter(y_resampled))
    
    # 3. Augmentation des données (Data Augmentation)
    print("Augmentation des données...")
    X_augmented = augment_time_series(X_resampled)
    X_final = np.concatenate([X_resampled, X_augmented])
    y_final = np.concatenate([y_resampled, y_resampled])
    print(f"Forme des données finales après augmentation: {X_final.shape}")
    
    # 4. One-hot encoding et Class Weights
    y_cat = to_categorical(y_final, num_classes=NUM_CLASSES)
    
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_resampled),
        y=y_resampled # Utiliser les labels non augmentés pour le calcul des poids
    )
    class_weights = dict(enumerate(class_weights))
    print("Class Weights:", class_weights)
    
    return X_final, y_final, y_cat, class_weights, X_array.shape[1:]


def split_data(X_final, y_final, y_cat):
    """
    Sépare les données en ensembles d'entraînement et de validation.
    """
    print("--- 2. SÉPARATION DES DONNÉES ---")
    X_train, X_val, y_train, y_val = train_test_split(
        X_final, y_cat, 
        test_size=TEST_SIZE,
        stratify=y_final, # Stratification basée sur les labels non encodés
        random_state=RANDOM_SEED
    )
    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")
    return X_train, X_val, y_train, y_val


def train_model(model, X_train, y_train, X_val, y_val, class_weights, callbacks):
    """
    Entraîne le modèle avec les données et les callbacks spécifiés.
    """
    print("--- 3. ENTRAÎNEMENT DU MODÈLE ---")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Sauvegarde finale du modèle (peut être le même que le meilleur checkpoint)
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    model.save(os.path.join(MODELS_PATH, "parkinson_cnn.keras"))
    print(f"Modèle sauvegardé dans {MODELS_PATH}/parkinson_cnn.keras")
    
    return history


def evaluate_model(model, X_val, y_val):
    """
    Évalue le modèle sur l'ensemble de validation et affiche le rapport.
    """
    print("\n--- 4. ÉVALUATION DU MODÈLE ---")
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)

    print("\nRapport de classification sur l'ensemble de validation :")
    print(classification_report(y_true, y_pred_classes, target_names=["Contrôle", "Parkinson"]))
    
    return y_pred, y_pred_classes, y_true


def plot_training_curves(history):
    """
    Trace et sauvegarde les courbes d'Accuracy et de Loss.
    """
    print("\n--- 5. TRACÉ DES COURBES ET RÉSULTATS FINAUX ---")
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
        
    plt.figure(figsize=(10,5))

    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.plot(history.history['loss'], label='Train Loss', linestyle='--', color='green')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', color='red')

    plt.xlabel("Epochs")
    plt.ylabel("Valeur")
    plt.title("Courbes Accuracy & Loss (CNN Parkinson) - MODÈLE CORRIGÉ")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(RESULTS_PATH, "training_curves.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Courbes sauvegardées dans {plot_path}")

    # Affichage des précisions finales
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\n=== RÉSULTATS FINAUX ===")
    print(f"Accuracy entraînement: {final_train_acc:.4f}")
    print(f"Accuracy validation: {final_val_acc:.4f}")
    print(f"Différence: {final_train_acc - final_val_acc:.4f} (Idéalement < 0.1 pour une bonne généralisation)")
    
# ====================================================================
#                                 MAIN
# ====================================================================

if __name__ == '__main__':
    # 1. Chargement, prétraitement et préparation des données
    X_final, y_final, y_cat, class_weights, input_shape_tuple = load_and_preprocess_data()

    # 2. Séparation des données
    X_train, X_val, y_train, y_val = split_data(X_final, y_final, y_cat)

    # 3. Création et compilation du modèle
    model = create_simplified_model(input_shape=input_shape_tuple)
    print("\n--- SOMMAIRE DU MODÈLE ---")
    model.summary()
    
    # 4. Configuration des Callbacks
    callbacks = setup_callbacks()

    # 5. Entraînement
    history = train_model(model, X_train, y_train, X_val, y_val, class_weights, callbacks)

    # 6. Évaluation finale (en utilisant le modèle avec les meilleurs poids restaurés par EarlyStopping)
    evaluate_model(model, X_val, y_val)

    # 7. Tracé
    plot_training_curves(history)
