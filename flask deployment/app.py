from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the ML model
model = joblib.load('models/Logistic_Regression.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        # Extract features and convert to float
        input_features = [float(data.get(x, 0)) for x in ['radius_mean', 'concavity_mean', 'smoothness_mean', 'texture_mean']]
        

        # Ensure valid input
        if not any(input_features):
            return jsonify({'error': 'Invalid input values'}), 400

        # Prepare input data for model
        df = pd.DataFrame([input_features], columns=['radius_mean', 'concavity_mean', 'smoothness_mean', 'texture_mean'])
        
        # Model prediction
        output = model.predict(df)
        res_val = "Malignant" if output[0] == 1 else "Benign"

        return jsonify({'prediction_text': f'Patient has {res_val}'})
    except Exception as e:

        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
