import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocess_signals import read_signals, normalize_signals, pad_signals
from load_data import load_demographics, get_file_labels
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split

# --- 1. Définir chemins ---
project_path = os.path.dirname(os.path.abspath(__file__))  # dossier scripts/
data_path = os.path.join(project_path, "../data")          # dossier data
model_path = os.path.join(project_path, "../models/best_model.keras")
model_path = os.path.abspath(model_path)  # chemin absolu

print("Chemin du modèle :", model_path)

# --- 2. Charger les données ---
demographics = load_demographics(data_path)
all_files, file_labels = get_file_labels(data_path, demographics)

X, y = read_signals(all_files, file_labels)
X_norm = normalize_signals(X)
X_array = pad_signals(X_norm)
y_array = np.array(y)

print("Forme des données :", X_array.shape)
print("Forme des labels :", y_array.shape)
print("Étiquettes uniques :", np.unique(y_array))

# --- 3. Charger le modèle ---
best_model = load_model(model_path)
print("Modèle chargé avec succès !")
print("Shape attendu par le modèle :", best_model.input_shape)

# Vérification compatibilité
if X_array.shape[1:] != best_model.input_shape[1:]:
    raise ValueError(
        f"Dimensions des données {X_array.shape[1:]} incompatibles avec le modèle {best_model.input_shape[1:]}"
    )

# --- 4. Split des données pour évaluation ---
X_train, X_test, y_train, y_test = train_test_split(
    X_array, y_array, 
    test_size=0.2, 
    stratify=y_array,
    random_state=42
)

# --- 5. Prédictions complètes ---
y_pred_probs = best_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# --- 6. Matrice de Confusion ---
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None):
    """
    Cette fonction affiche la matrice de confusion.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.title(title or 'Matrice de Confusion')
    plt.ylabel('Vraies étiquettes')
    plt.xlabel('Prédictions')
    plt.show()

# Affichage des deux versions
plot_confusion_matrix(y_test, y_pred, classes=['Contrôle', 'Parkinson'], 
                     title='Matrice de Confusion (Valeurs brutes)')
plot_confusion_matrix(y_test, y_pred, classes=['Contrôle', 'Parkinson'], 
                     normalize=True, title='Matrice de Confusion Normalisée')

# --- 7. Rapport de classification ---
print("\nRapport de Classification:")
print(classification_report(y_test, y_pred, target_names=['Contrôle', 'Parkinson']))

# --- 8. Prédire quelques exemples (visualisation) ---
num_samples = min(10, len(X_test))
sample_X = X_test[:num_samples]
sample_y_true = y_test[:num_samples]
sample_y_pred = y_pred[:num_samples]
sample_probs = y_pred_probs[:num_samples]

print("\nComparaison vraie étiquette vs prédiction :")
for i, (true, pred, probs) in enumerate(zip(sample_y_true, sample_y_pred, sample_probs)):
    print(f"Exemple {i+1} : Vrai = {true}, Prédit = {pred}, Probabilités = {probs}")

# --- 9. Graphique des prédictions ---
plt.figure(figsize=(10,4))
plt.bar(np.arange(num_samples) - 0.2, sample_y_true, width=0.4, label='Vrai')
plt.bar(np.arange(num_samples) + 0.2, sample_y_pred, width=0.4, label='Prédit')
plt.xlabel("Exemple")
plt.ylabel("Classe")
plt.title("Vrai vs Prédiction")
plt.legend()
plt.grid(True)
plt.show()

# --- 10. Tracer les signaux des exemples ---
for i in range(num_samples):
    plt.figure(figsize=(12,4))
    plt.title(f"Exemple {i+1} - Vrai: {sample_y_true[i]} / Prédit: {sample_y_pred[i]}\n"
              f"Probabilités: Contrôle={sample_probs[i][0]:.2f}, Parkinson={sample_probs[i][1]:.2f}")
    for j in range(X_array.shape[2]):
        plt.plot(sample_X[i,:,j], label=f"Capteur {j+1}")
    plt.xlabel("Temps")
    plt.ylabel("Amplitude normalisée")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# --- 11. Visualisation des erreurs ---
errors = np.where(y_pred != y_test)[0]
print(f"\nNombre d'erreurs: {len(errors)}/{len(y_test)} ({len(errors)/len(y_test):.1%})")

if len(errors) > 0:
    print("\nAnalyse des erreurs:")
    for i, error_idx in enumerate(errors[:3]):  # Affiche les 3 premières erreurs
        true_label = y_test[error_idx]
        pred_label = y_pred[error_idx]
        probs = y_pred_probs[error_idx]
        
        plt.figure(figsize=(12,4))
        plt.title(f"Erreur {i+1} - Vrai: {'Parkinson' if true_label else 'Contrôle'} | "
                 f"Prédit: {'Parkinson' if pred_label else 'Contrôle'}\n"
                 f"Probabilités: Contrôle={probs[0]:.2f}, Parkinson={probs[1]:.2f}")
        
        for j in range(X_array.shape[2]):
            plt.plot(X_test[error_idx,:,j], label=f"Capteur {j+1}")
        
        plt.xlabel("Temps")
        plt.ylabel("Amplitude normalisée")
        plt.legend()
        plt.grid(True)
        plt.show()