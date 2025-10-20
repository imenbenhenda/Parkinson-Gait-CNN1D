# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from load_data import load_demographics, get_file_labels
from preprocess_signals import read_signals, normalize_signals, unify_signal_length

# ==============================================================================
# 2. CONSTANTS AND CONFIGURATION
# ==============================================================================
# Define main paths
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "../data")
MODEL_PATH = os.path.join(PROJECT_PATH, "../models/best_model.keras")

# Define class names for plots
CLASS_NAMES = ['Control', 'Parkinson']
TARGET_SIGNAL_LENGTH = 1000  

# ==============================================================================
# 3. UTILITY AND ANALYSIS FUNCTIONS
# ==============================================================================

def load_and_preprocess_data(data_path):
    """Load and preprocess data for model evaluation."""
    print("--- Step 1: Loading and Preprocessing Data ---")
    demographics = load_demographics(data_path)
    all_files, file_labels = get_file_labels(data_path, demographics)

    X, y = read_signals(all_files, file_labels)
    X_norm = normalize_signals(X)
    X_array = unify_signal_length(X_norm, target_length=TARGET_SIGNAL_LENGTH)  
    y_array = np.array(y)
    
    print(f"Data shape after preprocessing: {X_array.shape}")
    print("Loading completed.\n")
    return X_array, y_array


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None):
    """Display a visually clear confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title or 'Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()


def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate model performance and display confusion matrix and classification report."""
    print("--- Step 3: Model Evaluation ---")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Display confusion matrices
    plot_confusion_matrix(y_test, y_pred, classes=class_names, title='Confusion Matrix (Raw Values)')
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True, title='Normalized Confusion Matrix')

    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("Evaluation completed.\n")


def visualize_errors(model, X_test, y_test, class_names):
    """Identify and visualize misclassified signals."""
    print("--- Step 4: Error Analysis ---")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    errors = np.where(y_pred != y_test)[0]
    print(f"Number of errors: {len(errors)}/{len(y_test)} ({len(errors)/len(y_test):.1%})")

    if len(errors) > 0:
        print("Displaying the first 3 misclassified samples...")
        for i, error_idx in enumerate(errors[:3]):
            true_label = y_test[error_idx]
            pred_label = y_pred[error_idx]
            probs = y_pred_probs[error_idx]
            
            plt.figure(figsize=(12, 4))
            plt.title(
                f"Error {i+1} - True: {class_names[true_label]} | Predicted: {class_names[pred_label]}\n"
                f"Probabilities: {class_names[0]}={probs[0]:.2f}, {class_names[1]}={probs[1]:.2f}"
            )
            for j in range(X_test.shape[2]):
                plt.plot(X_test[error_idx, :, j], label=f"Sensor {j+1}")
            
            plt.xlabel("Time")
            plt.ylabel("Normalized Amplitude")
            plt.legend()
            plt.grid(True)
            plt.show()
    print("Error analysis completed.\n")

# ==============================================================================
# 4. MAIN SCRIPT
# ==============================================================================

def main():
    """Main function to orchestrate data loading, evaluation, and visualization."""
    
    # Step 1: Load and preprocess data
    X_array, y_array = load_and_preprocess_data(DATA_PATH)
    
    # Step 2: Load the trained model
    print("--- Step 2: Loading Trained Model ---")
    print(f"Model path: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("Model successfully loaded!")
    
    # Verify shape compatibility
    if X_array.shape[1:] != model.input_shape[1:]:
        raise ValueError(
            f"Data dimensions {X_array.shape[1:]} are incompatible with model input {model.input_shape[1:]}"
        )
    print("Input shape verified.\n")
    
    # Step 3: Split data for evaluation (no retraining)
    _, X_test, _, y_test = train_test_split(
        X_array, y_array, test_size=0.2, stratify=y_array, random_state=42
    )

    # Step 4: Evaluate the model
    evaluate_model(model, X_test, y_test, CLASS_NAMES)
    
    # Step 5: Visualize misclassified samples
    visualize_errors(model, X_test, y_test, CLASS_NAMES)

if __name__ == '__main__':
    main()