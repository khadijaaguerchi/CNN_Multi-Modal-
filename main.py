# =============================================================================
# Author: Khadija Aguerchi, Younes Jabrane, Maryam Habba, Mustapha Ameur, Amir Hajjam El Hassani
# Affiliation: Cadi Ayyad University/ ENSA Marrakech / Modeling and Complex Systems (LMSC)
# Description: CNN-based breast cancer classification model
# Date: 2025-08-27
# =============================================================================


import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set random seed for reproducibility
tf.random.set_seed(42)

# Load dataset paths
data_dir = '/kaggle/input/breakhis-total/all'
benign_dir = os.path.join(data_dir, 'benign')
malignant_dir = os.path.join(data_dir, 'malignant')

# Function to load and preprocess images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skipping invalid image: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

# Load benign and malignant images
benign_images, benign_labels = load_images_from_folder(benign_dir, 0)
malignant_images, malignant_labels = load_images_from_folder(malignant_dir, 1)

# Combine data and labels
X = np.concatenate((benign_images, malignant_images), axis=0)
y = np.concatenate((benign_labels, malignant_labels), axis=0)

# Normalize pixel values
X = X / 255.0

# Cross-validation setup
num_runs = 3
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize lists to store metrics
run_fold_accuracies = []
run_metrics = []
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []
all_y_true = []
all_y_pred = []
total_execution_time = 0

# Function to normalize epoch lengths
def normalize_epoch_lengths(metric_lists):
    max_epochs = max(len(metrics) for metrics in metric_lists)
    normalized = [metrics + [metrics[-1]] * (max_epochs - len(metrics)) for metrics in metric_lists]
    return np.array(normalized)

# Run the model multiple times
for run in range(num_runs):
    print(f"\nRun {run+1}/{num_runs}")
    fold_accuracies = []
    metrics_per_run = []

    # KFold cross-validation
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        start_time = time.time()

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True
        )
        datagen.fit(X_train)

        # Define the model
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Callbacks
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint_filepath = f"model_run{run}_fold{fold}.keras"
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )

        # Train the model
        history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                            epochs=40,
                            validation_data=(X_test, y_test),
                            verbose=1,
                            callbacks=[early_stopping_callback, model_checkpoint_callback])

        # Store metrics
        train_accuracies.append(history.history['accuracy'])
        val_accuracies.append(history.history['val_accuracy'])
        train_losses.append(history.history['loss'])
        val_losses.append(history.history['val_loss'])

        # Evaluate the model
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        fold_accuracy = accuracy_score(y_test, y_pred)
        fold_precision = precision_score(y_test, y_pred)
        fold_recall = recall_score(y_test, y_pred)
        fold_f1 = f1_score(y_test, y_pred)

        print(f"Run {run+1}, Fold {fold}: Accuracy = {fold_accuracy * 100:.2f}%")

        fold_accuracies.append(fold_accuracy * 100)
        metrics_per_run.append([fold_accuracy, fold_precision, fold_recall, fold_f1])

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix - Run {run+1}, Fold {fold}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

        execution_time = time.time() - start_time
        total_execution_time += execution_time
        print(f"Execution time for Run {run+1}, Fold {fold}: {execution_time:.2f} seconds")

    run_fold_accuracies.append(fold_accuracies)
    run_metrics.append(metrics_per_run)

# Global confusion matrix
global_conf_matrix = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(global_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Global Confusion Matrix Across All Folds")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Convert accuracy results to DataFrame
columns = [f'Fold-{i+1}' for i in range(num_folds)]
df_results = pd.DataFrame(run_fold_accuracies, columns=columns)
df_results.index = [f'Run {i+1}' for i in range(num_runs)]
df_results['Avg.Acc'] = df_results.mean(axis=1)
df_results.loc['Final Avg'] = df_results.mean()

# Prepare metrics DataFrame
metrics_columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metrics_df = pd.DataFrame(columns=metrics_columns)
for run in range(num_runs):
    for fold in range(num_folds):
        metrics_df.loc[f'Run {run+1}, Fold {fold+1}'] = run_metrics[run][fold]

avg_metrics = metrics_df.groupby(metrics_df.index.str.extract(r'Run (\d)')[0]).mean()
avg_metrics.index = [f'Run {i}' for i in avg_metrics.index]
metrics_df = pd.concat([metrics_df, avg_metrics])
final_avg = metrics_df.mean()
final_avg_df = pd.DataFrame(final_avg).T
final_avg_df.index = ['Final Avg']
metrics_df = pd.concat([metrics_df, final_avg_df])

print("\nMetrics Evaluation Table:")
print(metrics_df)

# Plot training and validation accuracy
avg_train_accuracy = np.mean(normalize_epoch_lengths(train_accuracies), axis=0)
avg_val_accuracy = np.mean(normalize_epoch_lengths(val_accuracies), axis=0)

plt.figure(figsize=(10, 5))
plt.plot(avg_train_accuracy, label='Average Training Accuracy')
plt.plot(avg_val_accuracy, label='Average Validation Accuracy')
plt.title("Average Training and Validation Accuracy Across Folds")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation loss
avg_train_loss = np.mean(normalize_epoch_lengths(train_losses), axis=0)
avg_val_loss = np.mean(normalize_epoch_lengths(val_losses), axis=0)

plt.figure(figsize=(10, 5))
plt.plot(avg_train_loss, label='Average Training Loss')
plt.plot(avg_val_loss, label='Average Validation Loss')
plt.title("Average Training and Validation Loss Across Folds")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Print total execution time
# print(f"\nTotal Execution Time: {total_execution_time:.2f} seconds")


