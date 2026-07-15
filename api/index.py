"""
index.py
--------
Serverless Flask backend for Vercel deployment.
Loads precomputed model parameters, splits, and metrics from a JSON file,
and uses standard python libraries (csv/json) to serve the endpoints.
This removes all heavy production dependencies like pandas, numpy, and scikit-learn.
"""

from flask import Flask, jsonify, request
import os
import json
import csv

app = Flask(__name__, static_folder='../public', static_url_path='')

# Load dataset and train details
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, '..', 'data', 'model_results.json')
csv_path = os.path.join(script_dir, '..', 'data', 'student_scores.csv')

# Load the JSON results
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        model_results = json.load(f)
else:
    # Fail-safe hardcoded fallback parameters if training has not been run
    model_results = {
        'slope': 9.83557342813589,
        'intercept': 2.440578848123519,
        'r2': 0.9709419616099507,
        'mae': 3.370220689400262,
        'mse': 17.007358763595603,
        'rmse': 4.1240003350625,
        'train_hours': [],
        'train_scores': [],
        'test_hours': [],
        'test_scores': []
    }

# Read full dataset using built-in csv module (no pandas needed)
full_hours = []
full_scores = []
if os.path.exists(csv_path):
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None) # skip header
            for row in reader:
                if len(row) >= 2:
                    full_hours.append(float(row[0]))
                    full_scores.append(int(row[1]) if row[1].isdigit() else float(row[1]))
    except Exception as e:
        print(f"Error loading CSV file: {e}")

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

@app.route('/api/model')
def get_model():
    """Returns the trained model formula coefficients and key metrics."""
    return jsonify({
        'slope': model_results['slope'],
        'intercept': model_results['intercept'],
        'r2': model_results['r2'],
        'mae': model_results['mae'],
        'mse': model_results['mse'],
        'rmse': model_results['rmse']
    })

@app.route('/api/data')
def get_data():
    """Returns the split coordinates and full dataset lists for charting."""
    return jsonify({
        'hours': full_hours,
        'scores': full_scores,
        'train_hours': model_results['train_hours'],
        'train_scores': model_results['train_scores'],
        'test_hours': model_results['test_hours'],
        'test_scores': model_results['test_scores']
    })

@app.route('/api/predict')
def predict():
    """Predicts student marks using the precomputed regression equation."""
    try:
        hours_param = request.args.get('hours', 6.5)
        hours = float(hours_param)
        if not (0 <= hours <= 24):
            return jsonify({'error': 'Study hours must be between 0 and 24.'}), 400
            
        pred = model_results['intercept'] + (hours * model_results['slope'])
        pred_clipped = min(max(pred, 0), 100)
        return jsonify({
            'hours': hours,
            'prediction': float(pred_clipped),
            'raw_prediction': float(pred)
        })
    except ValueError:
        return jsonify({'error': 'Invalid numeric input for hours.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
