import numpy as np
import pandas as pd

def reduce_signal_length(X, target_length=1000):
    """
    Réduit la longueur des signaux pour éviter le sur-apprentissage
    """
    X_reduced = []
    for signal in X:
        if len(signal) > target_length:
            # Réduction simple : prendre les premiers target_length points
            reduced = signal[:target_length]
        else:
            # Si le signal est plus court, on le pad
            reduced = np.pad(signal, ((0, target_length - len(signal)), (0, 0)), mode='constant')
        X_reduced.append(reduced)
    return np.array(X_reduced)

def read_signals(files, file_labels):
    X, y = [], []
    for f in files:
        df = pd.read_csv(f, sep=r"\s+", header=None)
        signals = df.iloc[:, 1:17].values  # colonnes des capteurs
        X.append(signals)
        label = file_labels[f.name]
        y.append(0 if label=="CO" else 1)
    return X, y

def normalize_signals(X):
    X_norm = []
    for signals in X:
        min_vals = signals.min(axis=0)
        max_vals = signals.max(axis=0)
        norm = (signals - min_vals) / (max_vals - min_vals + 1e-6)
        X_norm.append(norm)
    return X_norm

def pad_signals(X_norm):
    max_len = max(signals.shape[0] for signals in X_norm)
    num_sensors = X_norm[0].shape[1]
    X_array = np.zeros((len(X_norm), max_len, num_sensors))
    for i, signals in enumerate(X_norm):
        length = signals.shape[0]
        X_array[i, :length, :] = signals
    return X_array
