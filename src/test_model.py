import os
import numpy as np
import collections
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam # Imported here for better clarity

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

from imblearn.over_sampling import RandomOverSampler

# --- Import external functions (Make sure they exist) ---
# NOTE: Code assumes 'load_data' and 'preprocess_signals' are available
from load_data import load_demographics, get_file_labels
from preprocess_signals import read_signals, normalize_signals, pad_signals, reduce_signal_length


# ====================================================================
#                     0. GLOBAL CONFIGURATION
# ====================================================================

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "../data")
MODELS_PATH = os.path.join(PROJECT_PATH, "../models")
RESULTS_PATH = os.path.join(PROJECT_PATH, "../results")

# Data Preparation Hyperparameters
TARGET_SIGNAL_LENGTH = 1000
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Model and Training Hyperparameters
NUM_CLASSES = 2
LEARNING_RATE = 0.0003
EPOCHS = 40
BATCH_SIZE = 16
PATIENCE_EARLY_STOP = 20
PATIENCE_REDUCE_LR = 15
MIN_LR = 1e-7

# ====================================================================
#                     1. PREPROCESSING & MODEL FUNCTIONS
# ====================================================================

def augment_time_series(signals, noise_factor=0.01):
    """
    Adds noise to signals to increase model robustness.
    """
    augmented = []
    for signal in signals:
        # Create normal noise and add it to the signal
        noise = np.random.normal(0, noise_factor, signal.shape)
        noisy_signal = signal + noise
        augmented.append(noisy_signal)
    return np.array(augmented)


def create_simplified_model(input_shape, num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE):
    """
    Defines and compiles the 1D CNN model for Parkinson's classification.
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # Block 1
        Conv1D(16, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),
        
        # Block 2
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
    Configures training callbacks (Checkpoint, Early Stopping, Reduce LR).
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
#                     2. MAIN PROCESSING FUNCTIONS
# ====================================================================

def load_and_preprocess_data():
    """
    Loads, normalizes, reduces and augments raw data.
    """
    print("--- 1. DATA LOADING AND PREPROCESSING ---")
    
    # 1. Initial loading and preprocessing
    demographics = load_demographics(DATA_PATH)
    all_files, file_labels = get_file_labels(DATA_PATH, demographics)
    X, y = read_signals(all_files, file_labels)
    X_norm = normalize_signals(X)
    X_array = pad_signals(X_norm)
    y_array = np.array(y)

    # Size reduction
    print(f"Shape before reduction: {X_array.shape}")
    X_array = reduce_signal_length(X_array, target_length=TARGET_SIGNAL_LENGTH)
    print(f"Shape after reduction: {X_array.shape}")
    print("Class distribution (initial):", collections.Counter(y_array))

    # 2. Oversampling (rebalancing)
    ros = RandomOverSampler(random_state=RANDOM_SEED)
    X_resampled, y_resampled = ros.fit_resample(X_array.reshape(len(X_array), -1), y_array)
    X_resampled = X_resampled.reshape(-1, X_array.shape[1], X_array.shape[2])
    print("After oversampling:", collections.Counter(y_resampled))
    
    # 3. Data augmentation
    print("Data augmentation...")
    X_augmented = augment_time_series(X_resampled)
    X_final = np.concatenate([X_resampled, X_augmented])
    y_final = np.concatenate([y_resampled, y_resampled])
    print(f"Final data shape after augmentation: {X_final.shape}")
    
    # 4. One-hot encoding and Class Weights
    y_cat = to_categorical(y_final, num_classes=NUM_CLASSES)
    
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_resampled),
        y=y_resampled # Use non-augmented labels for weight calculation
    )
    class_weights = dict(enumerate(class_weights))
    print("Class Weights:", class_weights)
    
    return X_final, y_final, y_cat, class_weights, X_array.shape[1:]


def split_data(X_final, y_final, y_cat):
    """
    Splits data into training and validation sets.
    """
    print("--- 2. DATA SPLITTING ---")
    X_train, X_val, y_train, y_val = train_test_split(
        X_final, y_cat, 
        test_size=TEST_SIZE,
        stratify=y_final, # Stratification based on non-encoded labels
        random_state=RANDOM_SEED
    )
    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")
    return X_train, X_val, y_train, y_val


def train_model(model, X_train, y_train, X_val, y_val, class_weights, callbacks):
    """
    Trains the model with specified data and callbacks.
    """
    print("--- 3. MODEL TRAINING ---")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Final model save (may be the same as best checkpoint)
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    model.save(os.path.join(MODELS_PATH, "parkinson_cnn.keras"))
    print(f"Model saved in {MODELS_PATH}/parkinson_cnn.keras")
    
    return history


def evaluate_model(model, X_val, y_val):
    """
    Evaluates the model on validation set and displays report.
    """
    print("\n--- 4. MODEL EVALUATION ---")
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)

    print("\nClassification report on validation set:")
    print(classification_report(y_true, y_pred_classes, target_names=["Control", "Parkinson"]))
    
    return y_pred, y_pred_classes, y_true


def plot_training_curves(history):
    """
    Plots and saves Accuracy and Loss curves.
    """
    print("\n--- 5. PLOTTING CURVES AND FINAL RESULTS ---")
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
        
    plt.figure(figsize=(10,5))

    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.plot(history.history['loss'], label='Train Loss', linestyle='--', color='green')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', color='red')

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Accuracy & Loss Curves (Parkinson CNN) - CORRECTED MODEL")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(RESULTS_PATH, "training_curves.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Curves saved in {plot_path}")

    # Display final accuracies
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\n=== FINAL RESULTS ===")
    print(f"Training accuracy: {final_train_acc:.4f}")
    print(f"Validation accuracy: {final_val_acc:.4f}")
    print(f"Difference: {final_train_acc - final_val_acc:.4f} (Ideally < 0.1 for good generalization)")
    
# ====================================================================
#                                 MAIN
# ====================================================================

if __name__ == '__main__':
    # 1. Data loading, preprocessing and preparation
    X_final, y_final, y_cat, class_weights, input_shape_tuple = load_and_preprocess_data()

    # 2. Data splitting
    X_train, X_val, y_train, y_val = split_data(X_final, y_final, y_cat)

    # 3. Model creation and compilation
    model = create_simplified_model(input_shape=input_shape_tuple)
    print("\n--- MODEL SUMMARY ---")
    model.summary()
    
    # 4. Callbacks configuration
    callbacks = setup_callbacks()

    # 5. Training
    history = train_model(model, X_train, y_train, X_val, y_val, class_weights, callbacks)

    # 6. Final evaluation (using model with best weights restored by EarlyStopping)
    evaluate_model(model, X_val, y_val)

    # 7. Plotting
    plot_training_curves(history)
