"""
generate_dataset.py
-------------------
This script generates a synthetic dataset representing the relationship
between the number of study hours and the corresponding student scores.
It is designed as a beginner-friendly demonstration of synthetic data generation.

Author: Antigravity
Date: 2026-07-15
"""

import os
import numpy as np
import pandas as pd

def generate_student_data():
    # Set a fixed random seed for reproducibility
    np.random.seed(42)
    
    # 1. Define dataset parameters
    num_students = 25
    base_score = 12.0  # Theoretical minimum score for 0 hours of study (intercept)
    rate_per_hour = 8.8  # Increase in score per hour of study (slope)
    noise_std = 5.0      # Standard deviation of the Gaussian noise representing external factors
    
    print("Generating synthetic student marks dataset...")
    
    # 2. Generate feature: Study Hours
    # Uniformly distribute hours between 1.0 and 10.0, rounded to 1 decimal place
    hours = np.round(np.random.uniform(1.0, 10.0, num_students), 1)
    
    # 3. Generate target: Scores with a linear relationship + Gaussian noise
    noise = np.random.normal(loc=0.0, scale=noise_std, size=num_students)
    scores = base_score + (hours * rate_per_hour) + noise
    
    # 4. Post-process scores: Clip to range [0, 100] and round to integers
    scores = np.clip(np.round(scores), 0, 100).astype(int)
    
    # 5. Create a Pandas DataFrame
    df = pd.DataFrame({
        'Hours': hours,
        'Scores': scores
    })
    
    # 6. Define output file path using os.path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    
    # Create the 'data' directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)
    
    csv_file_path = os.path.join(data_dir, 'student_scores.csv')
    
    # Save the dataframe to a CSV file
    df.to_csv(csv_file_path, index=False)
    
    print(f"Dataset successfully created and saved to: {os.path.abspath(csv_file_path)}")
    print("\nDataset Preview (First 5 rows):")
    print(df.head())
    print("\nDataset Summary Statistics:")
    print(df.describe())

if __name__ == "__main__":
    generate_student_data()
