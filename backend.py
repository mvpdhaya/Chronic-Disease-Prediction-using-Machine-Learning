# Backend: Flask API for Machine Learning Model
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib  # or pickle if joblib is not used

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for Flutter app

# Load the saved RandomForestClassifier model
MODEL_PATH = 'random_Forest_Classifier.pkl'  # Change path as needed
model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']  # Access the uploaded file
        data = pd.read_excel(file)  # Read Excel file into a DataFrame
        features = data.values  # Convert to numpy array
        predictions = model.predict(features).tolist()  # Predict for all rows
        results = ["Check doctor" if pred == 1 else "Good" for pred in predictions]
        return jsonify({"predictions": results})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True,port=5000)
