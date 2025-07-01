# Arrhythmia-detection-website
# 💓 ECG Arrhythmia Classifier

A full-stack deep learning web application that classifies ECG arrhythmia types from uploaded `.csv` files using a 1D Convolutional Neural Network (CNN). The model is trained on the UCI Arrhythmia Dataset and supports real-time predictions with a simple, responsive UI.


## 🚀 Features

- 🧠 **AI-Powered Classification** — Uses a trained 1D CNN model to detect arrhythmia types
- 📁 **CSV Upload** — Upload ECG `.csv` files (279 features) and get predictions instantly
- 📊 **Prediction Output** — Shows the **most probable class** and confidence score
- 💾 **Download Results** — Save predictions as CSV for further use or sharing
- 🌓 **Dark Mode UI** — Modern, responsive interface built with Tailwind CSS
- 🔍 **About & Results Pages** — Learn more about the model and view past predictions

---

## 🛠️ Tech Stack

### 🔧 **Backend**
| Tool | Purpose |
|------|---------|
| **Python 3** | Core programming language |
| **Flask** | Web framework to build REST API and serve frontend |
| **Flask-CORS** | Handles cross-origin requests between frontend and backend |
| **Pandas** | For reading ECG `.csv` files and manipulating data |
| **NumPy** | For numerical computations and array transformations |
| **scikit-learn** | For preprocessing pipeline: imputation, scaling, Random Forest feature selection |
| **imbalanced-learn (SMOTE)** | For balancing class distribution |
| **TensorFlow / Keras** | For building and serving the 1D CNN deep learning model |
| **joblib** | For loading saved preprocessing objects like scaler and selector |
| **Werkzeug** | For secure password hashing in login/register functionality |

### 🎨 **Frontend**
| Tool | Purpose |
|------|---------|
| **HTML5** | Structure of the web application |
| **Tailwind CSS** | Utility-first modern styling for responsive design |
| **Font Awesome** | Icon set used for UI elements (e.g., heart, upload, download) |
| **Google Fonts** | Poppins & Inter for smooth, clean typography |
| **Vanilla JavaScript** | Handles section switching, CSV upload, prediction rendering |

### 🧠 **Model & Data**
| Component | Details |
|-----------|---------|
| **Model** | 1D Convolutional Neural Network (CNN) built using Keras |
| **Dataset** | UCI Arrhythmia Dataset (279 features + class) |
| **Preprocessing** | Random Forest feature selection, SMOTE class balancing, StandardScaler & imputation (serialized with `joblib`) |

---

## 📁 File Upload Format

Make sure your `.csv` file:
- Has **exactly 279 numerical values**
- Has **no header**
- Does **not** include the class label

🧾 About

ECG Arrhythmia Classifier is an AI tool that predicts different types of heart arrhythmias from ECG signal data using deep learning.

It is trained on the UCI Arrhythmia Dataset with preprocessing steps like feature selection (Random Forest), scaling, imputation, and class balancing via SMOTE.
This tool is for educational and research purposes only — not for clinical use.
