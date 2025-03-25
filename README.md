# Breast Cancer Prediction System

![Screenshot from 2025-03-25 08-27-28](https://github.com/user-attachments/assets/13292f87-5516-4b58-89fb-4df34d7b57ae)

# Overview

This project is a Breast Cancer Prediction System built using logistic regression machine learning and deployed via Flask. The system classifies tumors as either Malignant or Benign based on extracted features from the dataset.

# Process: CRISP-DM Framework

This project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology:

## 1. Business Understanding

The goal is to develop a machine learning model that can predict whether a tumor is malignant or benign based on patient data.

This will assist doctors in diagnosing breast cancer more efficiently.

## 2. Data Understanding

The dataset consists of tumor characteristics such as radius_mean, texture_mean, smoothness_mean, and concavity_mean.

The target variable is Diagnosis (M = Malignant, B = Benign).

Initial exploration includes descriptive statistics and visualizations to understand data distribution.

## 3. Data Preparation

Feature Selection: Key features are selected based on domain knowledge.

Scaling: Standardization of numerical features using MinMaxScaler.

Handling Class Imbalance: SMOTE (Synthetic Minority Over-sampling Technique) is applied to balance the classes.

Train-Test Split: Data is split into 80% training and 20% testing.

## 4. Modeling

Three models were trained:

Logistic Regression

K-Nearest Neighbors (KNN)

Random Forest Classifier

Each model was evaluated using:

Accuracy Score

Classification Report

Confusion Matrix

Cross-validation Scores

Best-performing models were saved using joblib.

## 5. Evaluation

The Logistic Regression model achieved an accuracy of 96.4%.

Confusion matrix showed minimal false positives and false negatives.

Model performance was validated through cross-validation.

## 6. Deployment

The best model was deployed using Flask.

The trained model (Logistic_Regression.pkl) is loaded in the Flask app.

The API receives input features, processes them, and returns a prediction.

Output: "Patient has Malignant" or "Patient has Benign".

# Flask Deployment

## 1. Install Dependencies  

    pip install flask pandas numpy scikit-learn joblib  

## 2. Flask App Structure  

    ├── models/ </br>
    │   ├── Logistic_Regression.pkl </br>
    ├── app.py </br>
    ├── requirements.txt  

## 3. Run Flask Server  

    python app.py  

## 4. API Usage 

Send a POST request with tumor features:

    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"radius_mean": 17.99, "concavity_mean": 0.3, "smoothness_mean": 0.118, "texture_mean": 10.38}'

### Response:

Patient has Malignant

# Conclusion

This system successfully predicts breast cancer using machine learning, Flask API, and data preprocessing techniques. Future improvements could include deep learning models and real-time data integration.
