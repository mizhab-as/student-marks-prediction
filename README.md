# Student Marks Prediction Using Machine Learning

An AICTE AI Internship (Foundation Track, Beginner Mini Project level) submission-ready project demonstrating supervised machine learning concepts using Linear Regression.

---

## Project Description
This project predicts a student's marks based on the number of study hours. By utilizing the **Linear Regression** algorithm, the model learns from historical data (study hours paired with actual exam scores) to estimate the expected marks for new input values. The implementation serves as a clear, end-to-end demonstration of the primary phases of a Machine Learning workflow:
1. **Data Collection & Generation**: Creating a synthetic, labeled dataset of study hours and corresponding scores.
2. **Data Exploration & Visualisation**: Analyzing the dataset and plotting study hours against scores to understand their correlation.
3. **Data Preparation**: Dividing the dataset into training (80%) and testing (20%) sets to ensure fair model evaluation.
4. **Model Training**: Fitting a Linear Regression model on training data.
5. **Model Evaluation**: Making predictions on the testing data and evaluating performance using standard metrics (R², MAE, MSE, RMSE).
6. **Prediction**: Generating expected marks for preset study hours and providing an interactive prompt for live custom input.

---

## Technologies Used
- **Python**: Core programming language.
- **Pandas**: For loading, manipulating, and analyzing the dataset.
- **NumPy**: For mathematical computations and noise simulation.
- **Scikit-learn**: For splitting data (`train_test_split`), training the model (`LinearRegression`), and evaluating performance (`metrics`).
- **Matplotlib**: For plotting the exploratory scatter plot and the final regression fit line.

---

## Key Features
- **Dataset Generation**: Automatically generates a realistic synthetic dataset of 25 students with study hours between 1.0 and 10.0 hours.
- **Supervised ML Demonstration**: Showcases the fundamental concepts of supervised learning (mapping features to labels).
- **Comprehensive Evaluation**: Computes R² score, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
- **Dual Prediction Modes**: Offers predictions for preset sample study hours (2.5, 5, 7.5, 9.25 hours) and accepts live custom study hours via user input.
- **Robust Path Resolution**: Uses Python's `os.path` relative to `__file__` so that scripts run correctly from any directory.
- **Beginner-Friendly codebase**: Fully commented and cleanly organized pipeline with zero unnecessary abstraction.

---

## Learning Outcomes
- **Supervised Machine Learning Basics**: Understand how linear models map inputs (study hours) to continuous targets (scores).
- **Data Preprocessing**: Split data into training and validation sets to simulate testing on unseen data.
- **Regression Evaluation**: Interpret statistical performance indicators like R² and RMSE.
- **Analytical Graphing**: Create, label, and save correlation charts and model-fit lines to visually present model performance.

---

## Project Structure
```text
student_marks_prediction/
├── data/
│   └── student_scores.csv       # Generated dataset (25 student records)
├── outputs/
│   ├── 1_scatter_plot.png       # Correlation visualization (Hours vs Scores)
│   └── 2_regression_line.png    # Training/Testing fit line plot
├── src/
│   ├── generate_dataset.py      # Script to create data/student_scores.csv
│   └── marks_prediction.py      # ML pipeline training, evaluation, and inference
└── README.md                    # Project documentation and submission report
```

---

## How to Run

### 1. Install Dependencies
Ensure you have Python 3 installed. Run the following command to install the required libraries:
```bash
pip install pandas numpy matplotlib scikit-learn
```

### 2. Generate the Dataset
Create the synthetic student dataset by running:
```bash
python src/generate_dataset.py
```
This generates `data/student_scores.csv` containing study hours and score entries with Gaussian noise, and outputs a summary statistics overview.

### 3. Run the ML Pipeline & Predictions
Execute the regression model training and evaluation script:
```bash
python src/marks_prediction.py
```
*Note: In an interactive terminal, the script will prompt you to enter a custom number of study hours to output a prediction live.*

---

## Sample Results

### Learned Equation
The model fits the line:
$$\text{Marks} = 9.4028 \times \text{Hours} + 11.0074$$
*   **Slope ($m \approx 9.40$)**: For every additional hour a student studies, their score is predicted to increase by approximately 9.4%.
*   **Intercept ($c \approx 11.01$)**: A student studying for 0 hours is estimated to get a base score of 11.01%.

### Model Metrics on Test Set
- **R² Score (Coefficient of Determination)**: `0.9182` (indicates that ~91.8% of the variance in exam scores can be explained by study hours).
- **Mean Absolute Error (MAE)**: `5.6036` marks
- **Mean Squared Error (MSE)**: `39.5398` marks²
- **Root Mean Squared Error (RMSE)**: `6.2881` marks

### Test Set Actual vs Predicted Comparison Table
```text
 Hours  Actual Marks  Predicted Marks
   6.4            70             71.2
   3.7            38             45.8
   4.4            48             52.4
   4.3            42             51.4
   9.7            97            102.2
```

### Sample Predictions
- Studying for **2.50 hours/day** $\rightarrow$ predicted score: **34.51%**
- Studying for **5.00 hours/day** $\rightarrow$ predicted score: **58.02%**
- Studying for **7.50 hours/day** $\rightarrow$ predicted score: **81.53%**
- Studying for **9.25 hours/day** $\rightarrow$ predicted score: **97.98%**

---

## Possible Extensions
1. **Real-world Dataset**: Replace the synthetic dataset with a real-world dataset like the famous *Student Performance Dataset* from Kaggle.
2. **Multiple Linear Regression**: Incorporate additional variables such as class attendance, sleep hours, extra-curricular participation, or physical health to build a multiple regression model.
3. **Web Application Deployment**: Build a visual dashboard using **Streamlit** or **Flask** to allow users to adjust study hours via sliders and receive predictions dynamically in a web browser.
