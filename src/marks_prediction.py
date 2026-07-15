"""
marks_prediction.py
-------------------
This script builds a complete machine learning pipeline
to predict student marks based on their study hours. It demonstrates:
- Data exploration & visualization
- Training & testing split
- Linear Regression model fitting
- Model evaluation with standard regression metrics
- Interactive predictions for user input
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def run_pipeline():
    # Define directories relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    outputs_dir = os.path.join(script_dir, '..', 'outputs')
    
    # Create output directories if they do not exist
    os.makedirs(outputs_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, 'student_scores.csv')
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at '{csv_path}'. Please run generate_dataset.py first.")
        sys.exit(1)
        
    print("=" * 60)
    print("Step 1 - Data Collection")
    print("=" * 60)
    print(f"Loading dataset from: {os.path.abspath(csv_path)}")
    df = pd.read_csv(csv_path)
    print("Data loaded successfully.")
    
    print("\n" + "=" * 60)
    print("Step 2 - Data Exploration")
    print("=" * 60)
    print("\n--- First 5 rows of the dataset ---")
    print(df.head())
    
    print("\n--- Dataset Summary Statistics ---")
    print(df.describe())
    
    print("\n--- Checking for Missing Values ---")
    missing_vals = df.isnull().sum()
    print(missing_vals)
    
    # Save a scatter plot of Hours vs Scores
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Hours'], df['Scores'], color='blue', marker='o', label='Student Data')
    plt.title('Study Hours vs. Student Scores')
    plt.xlabel('Study Hours')
    plt.ylabel('Scores (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    scatter_plot_path = os.path.join(outputs_dir, '1_scatter_plot.png')
    plt.savefig(scatter_plot_path, dpi=300)
    plt.close()
    print(f"\nScatter plot successfully saved to: {os.path.abspath(scatter_plot_path)}")
    
    print("\n" + "=" * 60)
    print("Step 3 - Data Preparation")
    print("=" * 60)
    # Split into features (X) and target (y)
    X = df[['Hours']].values
    y = df['Scores'].values
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset split completed:")
    print(f"  - Training set size: {X_train.shape[0]} samples")
    print(f"  - Testing set size: {X_test.shape[0]} samples")
    
    print("\n" + "=" * 60)
    print("Step 4 - Model Training")
    print("=" * 60)
    # Fit the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    
    print("Model training complete.")
    print("\n--- Learned Linear Equation ---")
    print(f"Marks = {slope:.4f} * Hours + {intercept:.4f}")
    
    print("\n" + "=" * 60)
    print("Step 5 - Model Evaluation")
    print("=" * 60)
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Metrics calculations
    r2 = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print("\n--- Regression Metrics on Test Set ---")
    print(f"R² Score (Coefficient of Determination): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    print("\n--- Actual vs Predicted Comparison Table ---")
    comparison_df = pd.DataFrame({
        'Hours': X_test.flatten(),
        'Actual Marks': y_test,
        'Predicted Marks': np.round(y_pred, 1)
    })
    print(comparison_df.to_string(index=False))
    
    # Export parameters & splits to model_results.json
    import json
    model_results_path = os.path.join(data_dir, 'model_results.json')
    results = {
        'slope': float(slope),
        'intercept': float(intercept),
        'r2': float(r2),
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'train_hours': X_train.flatten().tolist(),
        'train_scores': y_train.tolist(),
        'test_hours': X_test.flatten().tolist(),
        'test_scores': y_test.tolist()
    }
    with open(model_results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nModel results successfully exported to JSON: {os.path.abspath(model_results_path)}")
    
    print("\n" + "=" * 60)
    print("Step 6 - Visualization")
    print("=" * 60)
    # Plot training points, testing points, and regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, color='blue', alpha=0.7, label='Training Data')
    plt.scatter(X_test, y_test, color='green', marker='s', alpha=0.9, label='Testing Data')
    
    # Plot regression line
    # Generate line points across the entire Hours domain [1, 10]
    x_line = np.linspace(1, 10, 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Regression Line (Marks = {slope:.2f}*Hours + {intercept:.2f})')
    
    plt.title('Linear Regression Model Fit')
    plt.xlabel('Study Hours')
    plt.ylabel('Scores (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    regression_plot_path = os.path.join(outputs_dir, '2_regression_line.png')
    plt.savefig(regression_plot_path, dpi=300)
    plt.close()
    print(f"Regression plot successfully saved to: {os.path.abspath(regression_plot_path)}")
    
    print("\n" + "=" * 60)
    print("Step 7 - Predictions on Unseen Data")
    print("=" * 60)
    # Predict for preset sample values
    sample_hours = [2.5, 5.0, 7.5, 9.25]
    print("\n--- Sample Predictions ---")
    for hours in sample_hours:
        pred_score = model.predict([[hours]])[0]
        # Clip to [0, 100] in case predictions exceed boundaries
        pred_score_clipped = np.clip(pred_score, 0, 100)
        print(f"Studying for {hours:4.2f} hours/day is predicted to yield a score of: {pred_score_clipped:.2f}%")
        
    print("\n--- Custom Live Prediction ---")
    try:
        # Check if running in an interactive terminal
        if sys.stdin.isatty():
            user_input = input("Enter custom study hours (between 0 and 24): ").strip()
            if user_input:
                try:
                    custom_hours = float(user_input)
                    if 0 <= custom_hours <= 24:
                        custom_pred = model.predict([[custom_hours]])[0]
                        custom_pred_clipped = np.clip(custom_pred, 0, 100)
                        print(f"\n--> Predicted score for {custom_hours:.2f} study hours is: {custom_pred_clipped:.2f}%")
                    else:
                        print("Error: Please enter a valid number of hours between 0 and 24.")
                except ValueError:
                    print("Error: Invalid numeric input.")
        else:
            print("Non-interactive environment detected. Skipping live prompt.")
    except (EOFError, KeyboardInterrupt):
        print("\nInteractive input interrupted. Skipping live prompt.")
    except Exception as e:
        print(f"Could not run live interactive prediction: {e}")

if __name__ == "__main__":
    run_pipeline()
