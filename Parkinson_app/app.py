from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import numpy as np
from Bio.PDB import PDBParser

# Step 1: Initialize Flask app
app = Flask(__name__)

# Step 2: Enable CORS
CORS(app)

# Step 3: Load model and scaler (ensure these .pkl files exist in the same folder)
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Step 4: Feature extraction function
def extract_features(pdb_path):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb_path)
    atoms = [atom for atom in structure.get_atoms() if atom.element != 'H']
    coords = np.array([atom.coord for atom in atoms])

    if coords.shape[0] == 0:
        return [0, 0, 0, 0]  # Return 4 values to match model

    centroid = np.mean(coords, axis=0)   # 3 values
    spread = np.std(coords)              # 1 scalar value

    # Now return only 4 features: x, y, z of centroid and overall spread
    return [centroid[0], centroid[1], centroid[2], spread]


# Step 5: Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        file_path = "temp.pdb"
        file.save(file_path)

        features = extract_features(file_path)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]

        result = "Positive" if prediction == 1 else "Negative"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": f"Error in prediction: {str(e)}"})

# Step 6: Run app
if __name__ == '__main__':
    app.run(debug=True)
