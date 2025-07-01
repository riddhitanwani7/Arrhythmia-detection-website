import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import json

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

def train():
    # Step 1: Load and clean dataset
    # Make sure 'arrhythmia.data' is in the 'model' directory
    data_path = 'model/arrhythmia.data'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please place arrhythmia.data in the 'model' directory.")
        return

    df = pd.read_csv(data_path, header=None)
    df.columns = [f'F{i}' for i in range(1, 280)] + ['class']
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df[~df['class'].isin([11, 12, 13])].reset_index(drop=True) # type: ignore

    # Step 3: Separate features and labels
    X = df.drop('class', axis=1)
    y = df['class'].values # type: ignore

    # Step 2: Impute missing values (on features only, to prevent data leakage)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Step 4: Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Step 5: Feature Selection - Top 89 features with RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    selector = SelectFromModel(rf, max_features=89, prefit=True)
    X_selected = selector.transform(X_scaled)
    print(f"Selected features shape: {X_selected.shape}") # type: ignore

    # Step 6: Apply SMOTE
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_resampled, y_resampled = smote.fit_resample(X_selected, y) # type: ignore
    print("After SMOTE class distribution:", np.bincount(y_resampled.astype(int)))

    # Create a mapping from original class labels to a new 0-indexed scheme
    unique_labels = np.unique(y_resampled)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    index_to_label = {i: str(label) for i, label in enumerate(unique_labels)} # Use str for JSON compatibility

    # Use the mapping to transform y labels
    y_indexed = np.array([label_to_index[label] for label in y_resampled])

    # Step 7: Reshape for CNN input
    X_cnn = X_resampled.reshape(X_resampled.shape[0], X_resampled.shape[1], 1)

    # One-hot encode labels
    y_encoded = to_categorical(y_indexed)

    # Step 8: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_cnn, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Step 9: Build CNN model
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_cnn.shape[1], 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(y_encoded.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Step 10: Train the model (no early stopping)
    history = model.fit(
        X_train, y_train,
        epochs=60,
        batch_size=32,
        validation_split=0.2
    )

    # Step 11: Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\n✅ Final Test Accuracy: {test_accuracy * 100:.2f}%")

    # Step 12: Classification report & Confusion matrix
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true_labels, y_pred_labels))
    
    # Step 13: Save model and preprocessors
    output_dir = 'model/model_files'
    os.makedirs(output_dir, exist_ok=True)

    model.save(os.path.join(output_dir, 'arrhythmia_cnn_model.h5'))
    joblib.dump(imputer, os.path.join(output_dir, 'imputer.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(selector, os.path.join(output_dir, 'selector.pkl'))
    with open(os.path.join(output_dir, 'label_mapping.json'), 'w') as f:
        json.dump(index_to_label, f)

    print(f"\n✅ Model, preprocessors, and label mapping saved to '{output_dir}' directory.")


    # Step 14: Plot Accuracy
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('CNN Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Step 15: Plot Loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('CNN Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train() 