"""
index.py
--------
Serverless Flask backend for Vercel deployment.
Loads the student scores dataset, trains the Linear Regression model,
and exposes API endpoints for model metrics, raw data, and predictions.

Author: Antigravity
Date: 2026-07-15
"""

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os

app = Flask(__name__, static_folder='../public', static_url_path='')

# Load dataset and train the regression model
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '..', 'data', 'student_scores.csv')

# Safe fallback dataset generation if file is missing (crucial for serverless environments)
if not os.path.exists(csv_path):
    np.random.seed(42)
    hours = np.round(np.random.uniform(1.0, 10.0, 25), 1)
    noise = np.random.normal(0, 5.0, 25)
    scores = np.clip(np.round(12.0 + (hours * 8.8) + noise), 0, 100).astype(int)
    df = pd.DataFrame({'Hours': hours, 'Scores': scores})
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
else:
    df = pd.read_csv(csv_path)

# Split and train model
X = df[['Hours']].values
y = df['Scores'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Calculate model stats
y_pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

slope = model.coef_[0]
intercept = model.intercept_

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

@app.route('/api/model')
def get_model():
    """Returns the trained model formula and key evaluation metrics."""
    return jsonify({
        'slope': float(slope),
        'intercept': float(intercept),
        'r2': float(r2),
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse)
    })

@app.route('/api/data')
def get_data():
    """Returns the train and test coordinates for charting."""
    return jsonify({
        'hours': df['Hours'].tolist(),
        'scores': df['Scores'].tolist(),
        'train_hours': X_train.flatten().tolist(),
        'train_scores': y_train.tolist(),
        'test_hours': X_test.flatten().tolist(),
        'test_scores': y_test.tolist()
    })

@app.route('/api/predict')
def predict():
    """Predicts student marks based on input study hours."""
    try:
        hours_param = request.args.get('hours', 6.5)
        hours = float(hours_param)
        if not (0 <= hours <= 24):
            return jsonify({'error': 'Study hours must be between 0 and 24.'}), 400
            
        pred = model.predict([[hours]])[0]
        pred_clipped = np.clip(pred, 0, 100)
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
