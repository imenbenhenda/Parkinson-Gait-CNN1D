# ======================================================================
# Parkinson Gait Classification - CNN Model Training
# ======================================================================
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, 
    BatchNormalization, Input
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from load_data import load_demographics, get_file_labels
from preprocess_signals import read_signals, normalize_signals, unify_signal_length

# ======================================================================
# Global Configuration
# ======================================================================

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "../data")
MODELS_PATH = os.path.join(PROJECT_PATH, "../models")
RESULTS_PATH = os.path.join(PROJECT_PATH, "../results")

TARGET_SIGNAL_LENGTH = 1000
RANDOM_SEED = 42
TEST_SIZE = 0.2
NUM_CLASSES = 2
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 60
PATIENCE_EARLY_STOP = 15
PATIENCE_REDUCE_LR = 10
MIN_LR = 1e-7

# ======================================================================
# Model Definition
# ======================================================================

def create_model(input_shape):
    """Build and compile the 1D CNN model."""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(16, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Dropout(0.3),
        GlobalAveragePooling1D(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def setup_callbacks():
    """Set up model training callbacks."""
    os.makedirs(MODELS_PATH, exist_ok=True)
    checkpoint = ModelCheckpoint(
        os.path.join(MODELS_PATH, "best_model.keras"),
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


def plot_training_curves(history):
    """Plot and save training and validation accuracy/loss curves."""
    os.makedirs(RESULTS_PATH, exist_ok=True)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(RESULTS_PATH, "training_curves.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Training curves saved to: {plot_path}")


# ======================================================================
# Main Script
# ======================================================================

if __name__ == '__main__':
    # Step 1: Load and preprocess data
    print("--- Step 1: Loading and preparing data ---")
    demographics = load_demographics(DATA_PATH)
    all_files, file_labels = get_file_labels(DATA_PATH, demographics)
    X, y = read_signals(all_files, file_labels)
    X_norm = normalize_signals(X)
    X_base = unify_signal_length(X_norm, target_length=TARGET_SIGNAL_LENGTH)
    y_base = np.array(y)

    print(f"Base data shape: {X_base.shape}")
    print(f"Initial class distribution: {collections.Counter(y_base)}")

    # Step 2: Split dataset
    print("\n--- Step 2: Splitting dataset ---")
    X_train, X_val, y_train, y_val = train_test_split(
        X_base, y_base,
        test_size=TEST_SIZE,
        stratify=y_base,
        random_state=RANDOM_SEED
    )

    # Step 3: Oversampling and data augmentation
    print("\n--- Step 3: Processing training data (Oversampling & Augmentation) ---")
    ros = RandomOverSampler(random_state=RANDOM_SEED)
    X_train_resampled, y_train_resampled = ros.fit_resample(
        X_train.reshape(len(X_train), -1), y_train
    )
    X_train_resampled = X_train_resampled.reshape(-1, TARGET_SIGNAL_LENGTH, X_base.shape[2])

    noise = np.random.normal(0, 0.01, X_train_resampled.shape)
    X_train_augmented = X_train_resampled + noise

    X_train_processed = np.concatenate([X_train_resampled, X_train_augmented])
    y_train_processed = np.concatenate([y_train_resampled, y_train_resampled])

    print(f"Processed training data shape: {X_train_processed.shape}")
    print(f"New training class distribution: {collections.Counter(y_train_processed)}")

    # Step 4: Prepare labels
    y_train_cat = to_categorical(y_train_processed, num_classes=NUM_CLASSES)
    y_val_cat = to_categorical(y_val, num_classes=NUM_CLASSES)

    # Step 5: Build and train model
    input_shape = (TARGET_SIGNAL_LENGTH, X_base.shape[2])
    model = create_model(input_shape)

    print("\n--- Model Summary ---")
    model.summary()

    callbacks = setup_callbacks()

    print("\n--- Step 4: Starting training ---")
    history = model.fit(
        X_train_processed, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Step 6: Evaluate model
    print("\n--- Step 5: Final evaluation on validation set ---")
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(y_val, y_pred_classes, target_names=["Control", "Parkinson"]))

    # Step 7: Plot training results
    plot_training_curves(history)
