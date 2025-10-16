import numpy as np
import pandas as pd
from typing import List

def read_signals(files: List[str], file_labels: dict) -> tuple[List[np.ndarray], List[int]]:
    """
    Reads signals from .txt files and converts them to numerical labels.

    Args:
        files (List[str]): List of file paths to read.
        file_labels (dict): Dictionary mapping filenames to their labels.

    Returns:
        tuple[List[np.ndarray], List[int]]: A list of signals and a list of labels (0 for CO, 1 for PT).
    """
    X, y = [], []
    for f in files:
        df = pd.read_csv(f, sep=r"\s+", header=None)
        signals = df.iloc[:, 1:17].values  # Columns for the 16 sensors
        X.append(signals)
        
        label = file_labels[f.name]
        y.append(0 if label == "CO" else 1)
        
    return X, y

def normalize_signals(X: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalizes each signal between 0 and 1 (Min-Max scaling).

    Args:
        X (List[np.ndarray]): List of signals to normalize.

    Returns:
        List[np.ndarray]: List of normalized signals.
    """
    X_norm = []
    for signals in X:
        min_vals = signals.min(axis=0)
        max_vals = signals.max(axis=0)
        # Add epsilon to avoid division by zero
        norm = (signals - min_vals) / (max_vals - min_vals + 1e-6)
        X_norm.append(norm)
    return X_norm

def pad_signals(X_norm: List[np.ndarray]) -> np.ndarray:
    """
    Adds padding to signals so they all have the same length.

    Args:
        X_norm (List[np.ndarray]): List of normalized signals.

    Returns:
        np.ndarray: A 3D numpy array with all padded signals.
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
    Reduces the length of each signal to a target size.

    Args:
        X (np.ndarray): 3D array of signals.
        target_length (int): The target length for each signal.

    Returns:
        np.ndarray: 3D array of signals with adjusted length.
    """
    # Uses numpy slicing, more efficient than a loop
    current_length = X.shape[1]
    
    if current_length > target_length:
        return X[:, :target_length, :]
    elif current_length < target_length:
        padding_needed = target_length - current_length
        return np.pad(X, ((0, 0), (0, padding_needed), (0, 0)), mode='constant')
    else:
        return X
