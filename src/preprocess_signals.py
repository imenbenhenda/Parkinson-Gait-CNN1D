import numpy as np
import pandas as pd
from typing import List

def read_signals(files: List[str], file_labels: dict) -> tuple[List[np.ndarray], List[int]]:
    """
    Lit les signaux depuis les fichiers .txt et les convertit en labels numériques.

    Args:
        files (List[str]): Liste des chemins des fichiers à lire.
        file_labels (dict): Dictionnaire associant les noms de fichiers à leurs labels.

    Returns:
        tuple[List[np.ndarray], List[int]]: Une liste de signaux et une liste de labels (0 pour CO, 1 pour PT).
    """
    X, y = [], []
    for f in files:
        df = pd.read_csv(f, sep=r"\s+", header=None)
        signals = df.iloc[:, 1:17].values  # Colonnes des 16 capteurs
        X.append(signals)
        
        label = file_labels[f.name]
        y.append(0 if label == "CO" else 1)
        
    return X, y

def normalize_signals(X: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalise chaque signal entre 0 et 1 (Min-Max scaling).

    Args:
        X (List[np.ndarray]): Liste des signaux à normaliser.

    Returns:
        List[np.ndarray]: Liste des signaux normalisés.
    """
    X_norm = []
    for signals in X:
        min_vals = signals.min(axis=0)
        max_vals = signals.max(axis=0)
        # Ajout d'un epsilon pour éviter la division par zéro
        norm = (signals - min_vals) / (max_vals - min_vals + 1e-6)
        X_norm.append(norm)
    return X_norm

def pad_signals(X_norm: List[np.ndarray]) -> np.ndarray:
    """
    Ajoute du padding aux signaux pour qu'ils aient tous la même longueur.

    Args:
        X_norm (List[np.ndarray]): Liste des signaux normalisés.

    Returns:
        np.ndarray: Un tableau numpy 3D avec tous les signaux paddés.
    """
    max_len = max(signals.shape[0] for signals in X_norm)
    num_sensors = X_norm[0].shape[1]
    
    X_array = np.zeros((len(X_norm), max_len, num_sensors))
    
    for i, signals in enumerate(X_norm):
        length = signals.shape[0]
        X_array[i, :length, :] = signals
        
    return X_array

def reduce_signal_length(X: np.ndarray, target_length: int = 1000) -> np.ndarray:
    """
    Réduit la longueur de chaque signal à une taille cible.

    Args:
        X (np.ndarray): Tableau 3D des signaux.
        target_length (int): La longueur cible pour chaque signal.

    Returns:
        np.ndarray: Tableau 3D des signaux avec la longueur ajustée.
    """
    # Utilise une découpe numpy, plus efficace qu'une boucle
    current_length = X.shape[1]
    
    if current_length > target_length:
        return X[:, :target_length, :]
    elif current_length < target_length:
        padding_needed = target_length - current_length
        return np.pad(X, ((0, 0), (0, padding_needed), (0, 0)), mode='constant')
    else:
        return X