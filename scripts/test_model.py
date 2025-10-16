# ==============================================================================
# 1. IMPORTATIONS
# ==============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Importations depuis vos modules locaux
from load_data import load_demographics, get_file_labels
from preprocess_signals import read_signals, normalize_signals, pad_signals, reduce_signal_length

# ==============================================================================
# 2. CONSTANTES ET CONFIGURATION
# ==============================================================================
# Définir les chemins principaux une seule fois
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "../data")
MODEL_PATH = os.path.join(PROJECT_PATH, "../models/best_model.keras")

# Définir les noms des classes pour les graphiques
CLASS_NAMES = ['Contrôle', 'Parkinson']

# ==============================================================================
# 3. FONCTIONS UTILITAIRES ET D'ANALYSE
# ==============================================================================

def load_and_preprocess_data(data_path):
    """Charge et prépare les données pour l'évaluation."""
    print("--- Étape 1: Chargement et Prétraitement des Données ---")
    demographics = load_demographics(data_path)
    all_files, file_labels = get_file_labels(data_path, demographics)

    X, y = read_signals(all_files, file_labels)
    X_norm = normalize_signals(X)
    X_array = pad_signals(X_norm)
    
    # Étape cruciale : s'assurer que les données de test ont la même forme que les données d'entraînement
    X_array = reduce_signal_length(X_array, target_length=1000)
    y_array = np.array(y)
    
    print(f"Forme des données après prétraitement : {X_array.shape}")
    print("Chargement terminé.\n")
    return X_array, y_array


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None):
    """Affiche une matrice de confusion visuellement agréable."""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title or 'Matrice de Confusion')
    plt.ylabel('Vraies étiquettes')
    plt.xlabel('Prédictions')
    plt.show()


def evaluate_model(model, X_test, y_test, class_names):
    """Évalue le modèle, affiche la matrice de confusion et le rapport de classification."""
    print("--- Étape 3: Évaluation du Modèle ---")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Affichage des matrices de confusion
    plot_confusion_matrix(y_test, y_pred, classes=class_names, title='Matrice de Confusion (Valeurs brutes)')
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True, title='Matrice de Confusion Normalisée')

    # Affichage du rapport de classification
    print("\nRapport de Classification:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("Évaluation terminée.\n")


def visualize_errors(model, X_test, y_test, class_names):
    """Trouve et visualise les signaux des prédictions incorrectes."""
    print("--- Étape 4: Analyse des Erreurs ---")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    errors = np.where(y_pred != y_test)[0]
    print(f"Nombre d'erreurs: {len(errors)}/{len(y_test)} ({len(errors)/len(y_test):.1%})")

    if len(errors) > 0:
        print("Affichage des 3 premières erreurs de prédiction...")
        for i, error_idx in enumerate(errors[:3]):
            true_label = y_test[error_idx]
            pred_label = y_pred[error_idx]
            probs = y_pred_probs[error_idx]
            
            plt.figure(figsize=(12, 4))
            plt.title(
                f"Erreur {i+1} - Vrai: {class_names[true_label]} | Prédit: {class_names[pred_label]}\n"
                f"Probabilités: {class_names[0]}={probs[0]:.2f}, {class_names[1]}={probs[1]:.2f}"
            )
            for j in range(X_test.shape[2]):
                plt.plot(X_test[error_idx, :, j], label=f"Capteur {j+1}")
            
            plt.xlabel("Temps")
            plt.ylabel("Amplitude normalisée")
            plt.legend()
            plt.grid(True)
            plt.show()
    print("Analyse des erreurs terminée.\n")

# ==============================================================================
# 4. SCRIPT PRINCIPAL
# ==============================================================================

def main():
    """Fonction principale qui orchestre le chargement, l'évaluation et la visualisation."""
    
    # Étape 1 : Chargement et préparation des données
    X_array, y_array = load_and_preprocess_data(DATA_PATH)
    
    # Étape 2 : Chargement du modèle entraîné
    print("--- Étape 2: Chargement du Modèle Entraîné ---")
    print(f"Chemin du modèle : {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("Modèle chargé avec succès !")
    
    # Vérification de la compatibilité des dimensions
    if X_array.shape[1:] != model.input_shape[1:]:
        raise ValueError(
            f"Dimensions des données {X_array.shape[1:]} incompatibles avec le modèle {model.input_shape[1:]}"
        )
    print("Compatibilité des dimensions vérifiée.\n")
    
    # Division des données en ensembles de test et d'entraînement (pour l'évaluation)
    # Note : On ne ré-entraîne pas, on utilise juste X_test et y_test.
    _, X_test, _, y_test = train_test_split(
        X_array, y_array, test_size=0.2, stratify=y_array, random_state=42
    )

    # Étape 3 : Évaluation complète du modèle
    evaluate_model(model, X_test, y_test, CLASS_NAMES)
    
    # Étape 4 : Visualisation des erreurs
    visualize_errors(model, X_test, y_test, CLASS_NAMES)

if __name__ == '__main__':
    main()
