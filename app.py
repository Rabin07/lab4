from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pickle
import pandas as pd

# Load the trained model
with open('fish_weight_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.json
    input_data = pd.DataFrame([data])

    # Make a prediction
    prediction = model.predict(input_data)

    # Return the prediction as JSON
    return jsonify({'predicted_weight': prediction[0]})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)