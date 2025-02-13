import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.impute import SimpleImputer
import random

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# Load the data
folder_path = 'C:\\Users\\paula\\Documents\\MEGAsync\\JKU\\5Semester\\Practical_work\\code\\data'
features = np.load(os.path.join(folder_path, 'features.npy'))
labels = np.load(os.path.join(folder_path, 'labels.npy'))

# Encode string labels to integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Flatten X_train for SMOTE
X_train_flat = X_train.reshape(X_train.shape[0], -1)

# Handle missing values
imputer = SimpleImputer(strategy='mean')  # Impute with the mean
X_train_flat = imputer.fit_transform(X_train_flat)

# Resample the data using SMOTE
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)

# Reshape back for LSTM
X_train_resampled = X_train_flat.reshape(-1, X_train.shape[1], X_train.shape[2])


print("Resampled label distribution:", Counter(y_train))


# If you impute X_train with 'imputer'
X_test_flat = X_test.reshape(X_test.shape[0], -1)
X_test_flat = imputer.transform(X_test_flat)  # use the same imputer
X_test = X_test_flat.reshape(-1, X_test.shape[1], X_test.shape[2])


# Define the LSTM model
model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    LSTM(64, kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(len(np.unique(labels)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_resampled, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.2f}")
print(f"Test Loss: {test_loss:.2f}")

# Predictions
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Decode labels
y_pred_labels = label_encoder.inverse_transform(y_pred_classes)
y_true_labels = label_encoder.inverse_transform(y_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Classification Report
report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)

# Print classification report
print("\nClassification Report:")
print(report)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_, cmap="Blues")
plt.title("Normalized Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# Loss and Accuracy Curves
history_dict = history.history

plt.figure(figsize=(12, 4))

# Loss Curve
plt.subplot(1, 2, 1)
plt.plot(history_dict['loss'], label='Training Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(history_dict['accuracy'], label='Training Accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Per-Class Accuracy
print("\nPer-Class Accuracy:")
for i, class_name in enumerate(label_encoder.classes_):
    class_accuracy = cm[i, i] / cm[i].sum()
    print(f"    {class_name}: {class_accuracy:.2f}")

# Overall Accuracy
overall_accuracy = accuracy_score(y_test, y_pred_classes)
print(f"\nOverall Accuracy: {overall_accuracy:.2f}")