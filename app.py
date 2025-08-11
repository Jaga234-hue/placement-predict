from flask import Flask, request, render_template
import pickle
import numpy as np
import os

# Paths for model and scaler
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

# Load the trained model
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Load the scaler used during training
with open(SCALER_PATH, 'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from the form
        features = [float(x) for x in request.form.values()]

        # Apply scaling (must match training)
        scaled_features = scaler.transform([features])

        # Make prediction
        prediction = model.predict(scaled_features)
        output = 'Placed' if prediction[0] == 1 else 'Not Placed'

        return render_template('index.html', prediction_text=f'Prediction: {output}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
