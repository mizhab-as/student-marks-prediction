"""
generate_dataset.py
-------------------
This script prepares the student scores dataset.
If a real dataset is found in the project path, it copies and prepares it.
Otherwise, it generates a synthetic dataset as a fallback.
"""

import os
import shutil
import numpy as np
import pandas as pd

def generate_student_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    archive_path = os.path.join(script_dir, '..', 'archive', 'score_updated.csv')
    data_dir = os.path.join(script_dir, '..', 'data')
    csv_file_path = os.path.join(data_dir, 'student_scores.csv')
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if the user's real dataset is available in the archive
    if os.path.exists(archive_path):
        print(f"Loading real dataset from archive: {os.path.abspath(archive_path)}")
        df = pd.read_csv(archive_path)
        # Standardize columns if necessary
        df.columns = [col.strip().capitalize() for col in df.columns]
        df.to_csv(csv_file_path, index=False)
        print(f"Real dataset copied and saved to: {os.path.abspath(csv_file_path)}")
    else:
        # Fallback to generating synthetic data
        print("Real dataset archive not found. Generating synthetic dataset fallback...")
        np.random.seed(42)
        num_students = 25
        base_score = 12.0
        rate_per_hour = 8.8
        noise_std = 5.0
        
        hours = np.round(np.random.uniform(1.0, 10.0, num_students), 1)
        noise = np.random.normal(loc=0.0, scale=noise_std, size=num_students)
        scores = np.clip(np.round(base_score + (hours * rate_per_hour) + noise), 0, 100).astype(int)
        
        df = pd.DataFrame({
            'Hours': hours,
            'Scores': scores
        })
        df.to_csv(csv_file_path, index=False)
        print(f"Synthetic dataset fallback saved to: {os.path.abspath(csv_file_path)}")
        
    print("\nDataset Preview (First 5 rows):")
    print(df.head())
    print("\nDataset Summary Statistics:")
    print(df.describe())

if __name__ == "__main__":
    generate_student_data()
