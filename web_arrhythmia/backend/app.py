from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import pandas as pd
import json
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'supersecretkey'  # Change this in production

# In-memory user store (username -> password_hash)
users = {}

CLASS_NAMES = {
    "1": "Normal",
    "2": "Ischemic changes (Coronary Artery Disease)",
    "3": "Old Anterior Myocardial Infarction",
    "4": "Old Inferior Myocardial Infarction",
    "5": "Sinus tachycardy",
    "6": "Sinus bradycardy",
    "7": "Ventricular Premature Contraction (PVC)",
    "8": "Supraventricular Premature Contraction",
    "9": "Left bundle branch block",
    "10": "Right bundle branch block",
    "14": "Pacing",
    "15": "Pacing",
    "16": "Unclassified"
}

# Define the path to the model and preprocessors
model_dir = os.path.join(os.path.dirname(__file__), '..', 'model', 'model_files')

# Load the model and preprocessors
try:
    model = load_model(os.path.join(model_dir, 'arrhythmia_cnn_model.h5'))
    imputer = joblib.load(os.path.join(model_dir, 'imputer.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    selector = joblib.load(os.path.join(model_dir, 'selector.pkl'))
    with open(os.path.join(model_dir, 'label_mapping.json'), 'r') as f:
        label_mapping = json.load(f)
    print("✅ Model, preprocessors, and mapping loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or preprocessors: {e}")
    model = None
    label_mapping = None

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    if username in users:
        return jsonify({'error': 'Username already exists'}), 400
    users[username] = generate_password_hash(password)
    return jsonify({'message': 'User registered successfully'})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if username in users and check_password_hash(users[username], password):
        session['user'] = username
        return jsonify({'message': 'Login successful'})
    return jsonify({'error': 'Invalid username or password'}), 401

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return jsonify({'message': 'Logged out'})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or label_mapping is None:
        return jsonify({'error': 'Model or label mapping is not loaded'}), 500
    if 'user' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    try:
        df = pd.read_csv(file, header=None) # type: ignore
        df.replace('?', np.nan, inplace=True)
        if df.shape[1] != 279:
            return jsonify({'error': f'Expected 279 columns, but got {df.shape[1]}'}), 400
        imputed_features = imputer.transform(df)
        scaled_features = scaler.transform(imputed_features)
        selected_features = selector.transform(scaled_features)
        cnn_input = selected_features.reshape((selected_features.shape[0], selected_features.shape[1], 1))
        predictions_array = model.predict(cnn_input)
        predicted_indices = np.argmax(predictions_array, axis=1)
        confidences = np.max(predictions_array, axis=1)
        results = []
        for i, index in enumerate(predicted_indices):
            class_number = label_mapping[str(index)]
            class_name = CLASS_NAMES.get(class_number, "Unknown Class")
            confidence = float(confidences[i])
            results.append({
                "class_number": class_number,
                "class_name": class_name,
                "confidence": confidence
            })
        return jsonify({'predictions': results})
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) 