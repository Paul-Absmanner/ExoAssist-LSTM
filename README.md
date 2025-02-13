# Predicting Human Movements for Exoskeleton Assistance

## Overview
This project focuses on **real-time motion classification** to enhance **exoskeleton-assisted leg support**. It utilizes **OptiTrack motion capture** data and **Long Short-Term Memory (LSTM) neural networks** to predict key movement transitions such as **walking, sitting down, and standing up**. The system is designed to enable **adaptive torque control**, ensuring smooth transitions between movement states while reducing **physical strain** on delivery personnel.

## Project Structure
The repository consists of two main components:

- **PreProcessing.py** – Data preprocessing and segmentation.
- **Training.py** – Model training, evaluation, and performance analysis.

---

## 1. Data Preprocessing (`PreProcessing.py`)

### **Functionality**
- Loads CSV files containing motion data (OptiTrack 16-marker system).
- Removes unnecessary columns (antenna data, rotation data).
- **Normalizes** coordinate values using MinMaxScaler.
- Segments data into **50 time-step sequences** with a **25 time-step overlap**.
- Assigns a label to each sequence based on the most frequent movement within that window.
- Handles missing values by imputing with the **mean of each feature column**.
- Saves preprocessed data as `.npy` files (`features.npy`, `labels.npy`, `file_times.npy`).

### **Output**
- `features.npy` – Time-series sequences (LSTM input).
- `labels.npy` – Encoded movement labels.
- `file_times.npy` – Metadata tracking sequence origins.

---

## 2. Model Training (`Training.py`)

### **Neural Network Architecture**
- **LSTM (128 units, return_sequences=True)** – Extracts time-dependent features.
- **Dropout (0.4)** – Prevents overfitting.
- **LSTM (64 units)** – Captures deeper sequential dependencies.
- **Dropout (0.4)** – Regularization.
- **Dense (32 units, ReLU activation)** – Further feature refinement.
- **Dense (Softmax activation, output classes = 3)** – Predicts movement category.

### **Training Process**
1. **Data Loading** – Reads `features.npy` and `labels.npy`.
2. **Label Encoding** – Converts movement labels to integers.
3. **Data Splitting** – 80% training, 20% testing (`train_test_split`).
4. **Imputation** – Missing values are filled with the mean (`SimpleImputer`).
5. **Resampling (Optional)** – SMOTE used for class balancing.
6. **Model Compilation** – `Adam` optimizer, `Sparse Categorical Crossentropy` loss.
7. **Training** – 15 epochs, batch size of 32.
8. **Evaluation** – Accuracy, loss, confusion matrix, classification report.

---

## 3. Model Evaluation & Results

### **Evaluation Metrics**
- **Accuracy** – Measures overall correctness of predictions.
- **Confusion Matrix** – Analyzes misclassifications across movement classes.
- **Precision, Recall, and F1-score** – Quantifies class-wise prediction performance.
- **Training & Validation Loss** – Identifies potential overfitting.
- **Per-Class Accuracy** – Assesses movement-specific model performance.

### **Visualization**
- **Confusion Matrix Plot** – Highlights correct vs. incorrect predictions.
- **Loss & Accuracy Curves** – Tracks model learning trends over epochs.

---

## 4. Installation & Usage

### **Requirements**
- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- Imbalanced-learn (for SMOTE)

### **Running the Code**
```bash
# Step 1: Data Preprocessing
python PreProcessing.py

# Step 2: Model Training & Evaluation
python Training.py
