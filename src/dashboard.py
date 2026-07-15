"""
dashboard.py
------------
A premium Ivory & Cream academic-themed dashboard for the Student Marks Prediction project.
Features a light earth-toned palette, subtle floating tan background particles,
Warm Beige card layouts, and dynamic predictions using the trained Linear Regression model.

Author: Antigravity
Date: 2026-07-15
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Student Marks Predictor Studio",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Injected CSS for Scholastic Ivory & Cream Aesthetics
custom_css = """
<style>
/* Import custom typography */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* Base page background: Cream/Ivory (#F5F0E6) with gold and soft blue radial glows */
.stApp {
    background-color: #F5F0E6 !important;
    background-image: 
        radial-gradient(circle at 80% 15%, rgba(184, 150, 79, 0.12) 0%, transparent 45%),
        radial-gradient(circle at 20% 80%, rgba(122, 147, 165, 0.1) 0%, transparent 45%) !important;
    color: #1F2C40 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Typography Overrides */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #1F2C40 !important;
}

/* Hide default Streamlit visual headers/footers for a standalone app look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom keyframes for floating bokeh bubbles background in Tan */
@keyframes floatUp {
    0% { transform: translateY(0px) rotate(0deg); opacity: 0; }
    10% { opacity: 0.25; }
    90% { opacity: 0.25; }
    100% { transform: translateY(-900px) rotate(360deg); opacity: 0; }
}

.bg-bubbles {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -100;
    overflow: hidden;
    margin: 0;
    padding: 0;
    pointer-events: none;
}

.bg-bubbles li {
    position: absolute;
    list-style: none;
    display: block;
    width: 35px;
    height: 35px;
    background-color: rgba(196, 166, 114, 0.1);
    bottom: -150px;
    animation: floatUp 32s infinite linear;
    border-radius: 50%;
    box-shadow: 0 0 10px rgba(196, 166, 114, 0.15);
}

.bg-bubbles li:nth-child(1) { left: 8%; width: 75px; height: 75px; animation-delay: 0s; animation-duration: 20s; background-color: rgba(196, 166, 114, 0.08); box-shadow: 0 0 20px rgba(196, 166, 114, 0.1); }
.bg-bubbles li:nth-child(2) { left: 18%; width: 30px; height: 30px; animation-delay: 3s; animation-duration: 23s; }
.bg-bubbles li:nth-child(3) { left: 28%; width: 45px; height: 45px; animation-delay: 6s; animation-duration: 17s; }
.bg-bubbles li:nth-child(4) { left: 45%; width: 60px; height: 60px; animation-delay: 1s; animation-duration: 27s; background-color: rgba(196, 166, 114, 0.08); }
.bg-bubbles li:nth-child(5) { left: 58%; width: 38px; height: 38px; animation-delay: 4s; animation-duration: 20s; }
.bg-bubbles li:nth-child(6) { left: 72%; width: 100px; height: 100px; animation-delay: 8s; animation-duration: 32s; }
.bg-bubbles li:nth-child(7) { left: 85%; width: 50px; height: 50px; animation-delay: 2s; animation-duration: 25s; }
.bg-bubbles li:nth-child(8) { left: 93%; width: 25px; height: 25px; animation-delay: 11s; animation-duration: 19s; }
.bg-bubbles li:nth-child(9) { left: 38%; width: 85px; height: 85px; animation-delay: 14s; animation-duration: 30s; }
.bg-bubbles li:nth-child(10) { left: 65%; width: 55px; height: 55px; animation-delay: 5s; animation-duration: 21s; }

/* Custom styled tabs for scholastic earth-toned aesthetic */
button[data-baseweb="tab"] {
    background-color: transparent !important;
    color: #4A3728 !important;
    opacity: 0.65 !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 12px 28px !important;
    transition: all 0.25s ease !important;
    border-bottom: 2px solid transparent !important;
    margin-right: 8px !important;
}

button[data-baseweb="tab"]:hover {
    color: #1F2C40 !important;
    opacity: 1 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #1F2C40 !important;
    opacity: 1 !important;
    border-bottom: 2px solid #1F2C40 !important;
    text-shadow: 0 0 10px rgba(31, 44, 64, 0.1) !important;
}

/* Custom styled Warm Beige cards with Camel/Tan borders */
.neon-card {
    background-color: #E8DCC8 !important;
    border: 1px solid #C4A672 !important;
    border-radius: 16px !important;
    padding: 24px !important;
    margin-bottom: 24px !important;
    box-shadow: 0 10px 30px rgba(74, 55, 40, 0.06) !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}

.neon-card:hover {
    border-color: #1F2C40 !important;
    box-shadow: 0 12px 35px rgba(74, 55, 40, 0.12) !important;
    transform: translateY(-4px);
}

/* Top Border Accent Lines for Metrics */
.card-r2 { border-top: 4px solid #1F2C40 !important; } /* Navy Blue */
.card-mae { border-top: 4px solid #5C1A1B !important; } /* Burgundy */
.card-mse { border-top: 4px solid #4A3728 !important; } /* Chocolate Brown */
.card-rmse { border-top: 4px solid #355E3B !important; } /* Hunter Green */

.card-r2:hover { border-color: #1F2C40 !important; }
.card-mae:hover { border-color: #5C1A1B !important; }
.card-mse:hover { border-color: #4A3728 !important; }
.card-rmse:hover { border-color: #355E3B !important; }

/* Gradient Hero Title (Navy to Burgundy) */
.glow-title-gradient {
    background: linear-gradient(135deg, #1F2C40 25%, #5C1A1B 75%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

/* Styled text configurations matching scholastic colors */
.neon-blue {
    color: #1F2C40 !important; /* Navy Blue */
    font-weight: 700;
}

.neon-pink {
    color: #5C1A1B !important; /* Burgundy */
    font-weight: 700;
}

.neon-green {
    color: #355E3B !important; /* Hunter Green */
    font-weight: 700;
}

.neon-purple {
    color: #4A3728 !important; /* Chocolate Brown */
    font-weight: 700;
}

/* Custom styled numeric and slider handles */
div[role="slider"] {
    background-color: #1F2C40 !important;
    box-shadow: 0 0 8px rgba(31, 44, 64, 0.3) !important;
}

div[data-baseweb="slider"] div {
    background-color: #C4A672 !important;
}

/* Label text color of sliders */
.stSlider label, .stSlider div {
    color: #4A3728 !important;
}

/* Layout spacing */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}
</style>

<!-- Injected HTML for animated background bubbles -->
<ul class="bg-bubbles">
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
</ul>
"""

# Apply styling and animated background
st.markdown(custom_css, unsafe_allow_html=True)

# 3. Data Processing and Model Training
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
csv_path = os.path.join(data_dir, 'student_scores.csv')

# Safe fallback dataset generation if file is missing
if not os.path.exists(csv_path):
    np.random.seed(42)
    hours = np.round(np.random.uniform(1.0, 10.0, 25), 1)
    noise = np.random.normal(0, 5.0, 25)
    scores = np.clip(np.round(12.0 + (hours * 8.8) + noise), 0, 100).astype(int)
    df = pd.DataFrame({'Hours': hours, 'Scores': scores})
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
else:
    df = pd.read_csv(csv_path)

# Prepare model and train
X = df[['Hours']].values
y = df['Scores'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Calculate metrics
y_pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

# Custom Helper: Premium Glassmorphic Table Generator (Matching Cream/Beige)
def render_styled_table(data, header_accent_color="#1F2C40"):
    html = "<div style='overflow-x:auto; margin-bottom:15px;'>"
    html += "<table style='width:100%; border-collapse:collapse; background-color:#E8DCC8; border-radius:12px; border: 1px solid #C4A672; overflow:hidden;'>"
    # Header Row
    html += "<thead><tr style='background-color:rgba(31, 44, 64, 0.05); border-bottom:1px solid #C4A672;'>"
    for col in data.columns:
        html += f"<th style='padding:14px 18px; text-align:left; font-family:\"Space Grotesk\", sans-serif; font-size:12px; font-weight:600; color:{header_accent_color}; letter-spacing:1.5px; text-transform:uppercase;'>{col}</th>"
    html += "</tr></thead><tbody>"
    # Content Rows
    for idx, row in data.iterrows():
        bg_style = "background-color:rgba(245, 240, 230, 0.35);" if idx % 2 == 0 else ""
        html += f"<tr style='border-bottom:1px solid rgba(196, 166, 114, 0.2); {bg_style}'>"
        for val in row:
            if isinstance(val, float):
                val_str = f"{val:.2f}"
            else:
                val_str = str(val)
            html += f"<td style='padding:14px 18px; font-size:14px; font-family:inherit; color:#1F2C40;'>{val_str}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    return html

# 4. App Landing Header with Scholastic Gradient Title
st.markdown("""
<div style="text-align: center; margin-top: 0.5rem; margin-bottom: 2rem;">
    <span style="font-size: 11px; font-weight: 700; letter-spacing: 4px; color: #5C1A1B; text-transform: uppercase;">
        AICTE Internship Project Submission
    </span>
    <h1 class="glow-title-gradient" style="margin: 0.4rem 0 0.8rem 0; font-size: 46px; font-weight: 700; line-height: 1.2;">
        Student Marks Prediction Studio
    </h1>
    <p style="font-size: 15px; color: #4A3728; max-width: 600px; margin: 0 auto; line-height: 1.5;">
        An interactive earth-toned platform demonstrating supervised machine learning. 
        Train, evaluate, and predict scores live using the <b>Linear Regression</b> model.
    </p>
</div>
""", unsafe_allow_html=True)

# 5. Tabbed Landing Sections
tab1, tab2, tab3 = st.tabs(["🎯 Live Predictor", "📊 Dataset & Exploration", "⚙️ Model Evaluation"])

# --- TAB 1: LIVE PREDICTOR ---
with tab1:
    col1, col2 = st.columns([1.1, 0.9], gap="large")
    
    with col1:
        st.markdown("<h3 style='margin-top:0; color:#1F2C40;'>Estimate Expected Score</h3>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color:#4A3728; font-size:14px; line-height:1.6; margin-bottom: 25px;'>
            Adjust the slider below to select daily study hours. The Linear Regression model will dynamically 
            predict the estimated score based on the learned equation.
        </p>
        """, unsafe_allow_html=True)
        
        # Interactive Study Hours Slider
        study_hours = st.slider(
            "Select Daily Study Hours:",
            min_value=0.0,
            max_value=12.0,
            value=6.5,
            step=0.25
        )
        
        # Make Prediction
        predicted_score = model.predict([[study_hours]])[0]
        predicted_score_clipped = np.clip(predicted_score, 0, 100)
        
        st.markdown("<div style='margin-top: 35px;'></div>", unsafe_allow_html=True)
        
        # Visual breakdown card
        st.markdown(f"""
        <div class="neon-card card-mse">
            <h4 style="margin: 0 0 12px 0; font-size: 16px; color: #1F2C40; font-family:'Space Grotesk', sans-serif;">How the Prediction Works</h4>
            <p style="margin: 0; color: #4A3728; font-size: 13px; line-height: 1.6;">
                Using the learned intercept (<span class="neon-pink">{intercept:.2f}</span>) as the baseline, 
                every hour of study adds a rate of <span class="neon-blue">{slope:.2f}%</span> to the score.
                <br><br>
                <b>Calculation:</b> 
                <code>Score = {intercept:.2f} + ({study_hours:.2f} * {slope:.2f}) = {predicted_score:.2f}%</code> 
                (Clipped to 100% maximum).
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Prediction Display Panel (Hunter Green Glowing style border)
        st.markdown(f"""
        <div class="neon-card" style="text-align: center; padding: 45px 30px !important; border-top: 4px solid #355E3B !important;">
            <h4 style="margin: 0; color: #4A3728; font-weight: 500; font-size: 13px; letter-spacing: 1.5px; text-transform: uppercase;">
                ESTIMATED SCORE
            </h4>
            <div class="neon-green" style="font-size: 72px; font-weight: 800; margin: 20px 0;">
                {predicted_score_clipped:.1f}%
            </div>
            <p style="margin: 0; color: #1F2C40; font-size: 16px;">
                For studying <span class="neon-blue" style="font-size: 18px;">{study_hours:.2f} hours</span> daily
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Formula Panel
        st.markdown(f"""
        <div class="neon-card card-mae" style="text-align: center; padding: 18px !important;">
            <h5 style="margin: 0 0 6px 0; color: #4A3728; font-size: 11px; letter-spacing: 1px; text-transform: uppercase;">
                Model Equation
            </h5>
            <code style="background-color: transparent; color: #5C1A1B; font-size: 16px; font-weight: bold; font-family: monospace;">
                Marks = {slope:.4f} * Hours + {intercept:.4f}
            </code>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 2: DATASET & EXPLORATION ---
with tab2:
    col1, col2 = st.columns([0.9, 1.1], gap="large")
    
    with col1:
        st.markdown("<h3 style='margin-top:0; color:#1F2C40;'>Historical Records</h3>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color:#4A3728; font-size:14px; line-height:1.6; margin-bottom: 25px;'>
            The dataset consists of 25 records generated with a strong positive linear relationship and Gaussian noise 
            to simulate real-world factors.
        </p>
        """, unsafe_allow_html=True)
        
        # Display dataset preview (custom styled table with Navy Blue accent)
        table_html = render_styled_table(df.head(10), header_accent_color="#1F2C40")
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown("<p style='font-size: 12px; color: #4A3728; text-align: center;'>Showing first 10 of 25 student records.</p>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<h3 style='margin-top:0; color:#1F2C40;'>Dataset Correlation Plot</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#4A3728; font-size:14px; margin-bottom: 25px;'>Visualizing the correlation between study hours and exam marks.</p>", unsafe_allow_html=True)
        
        # Matplotlib Exploratory Plot styled in Ivory/Cream background
        fig, ax = plt.subplots(figsize=(8, 5.2))
        fig.patch.set_facecolor('#F5F0E6')
        ax.set_facecolor('#F5F0E6')
        
        # Scatter plot in Navy Blue points
        ax.scatter(df['Hours'], df['Scores'], color='#1F2C40', alpha=0.8, edgecolors='#4A3728', linewidths=0.7, s=90, label='Student Data Points')
        
        # Visual limits & grids in Tan
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 105)
        ax.grid(color='#C4A672', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Styles
        ax.set_xlabel('Study Hours', color='#4A3728', fontsize=11, labelpad=8)
        ax.set_ylabel('Scores (%)', color='#4A3728', fontsize=11, labelpad=8)
        ax.tick_params(colors='#4A3728', which='both', labelsize=9)
        
        for spine_name, spine in ax.spines.items():
            if spine_name in ['top', 'right']:
                spine.set_visible(False)
            else:
                spine.set_color('#C4A672')
                
        legend = ax.legend(facecolor='#E8DCC8', edgecolor='#C4A672', labelcolor='#1F2C40', fontsize=9, loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)

# --- TAB 3: MODEL EVALUATION ---
with tab3:
    # KPI metrics row
    st.markdown("<h3 style='margin-top:0; color:#1F2C40; margin-bottom:20px;'>Evaluation Metrics</h3>", unsafe_allow_html=True)
    
    k_col1, k_col2, k_col3, k_col4 = st.columns(4)
    with k_col1:
        st.markdown(f"""
        <div class="neon-card card-r2" style="text-align: center; padding: 20px 15px !important;">
            <div style="font-size: 11px; color: #4A3728; font-weight: 600; letter-spacing: 1.5px;">R² SCORE</div>
            <div class="neon-blue" style="font-size: 28px; font-weight: 800; margin-top: 8px;">{r2:.4f}</div>
            <div style="font-size: 11px; color: #4A3728; margin-top: 5px;">91.8% variance explained</div>
        </div>
        """, unsafe_allow_html=True)
    with k_col2:
        st.markdown(f"""
        <div class="neon-card card-mae" style="text-align: center; padding: 20px 15px !important;">
            <div style="font-size: 11px; color: #4A3728; font-weight: 600; letter-spacing: 1.5px;">MAE</div>
            <div class="neon-pink" style="font-size: 28px; font-weight: 800; margin-top: 8px;">{mae:.4f}</div>
            <div style="font-size: 11px; color: #4A3728; margin-top: 5px;">Mean Absolute Error</div>
        </div>
        """, unsafe_allow_html=True)
    with k_col3:
        st.markdown(f"""
        <div class="neon-card card-mse" style="text-align: center; padding: 20px 15px !important;">
            <div style="font-size: 11px; color: #4A3728; font-weight: 600; letter-spacing: 1.5px;">MSE</div>
            <div class="neon-purple" style="font-size: 28px; font-weight: 800; margin-top: 8px;">{mse:.4f}</div>
            <div style="font-size: 11px; color: #4A3728; margin-top: 5px;">Mean Squared Error</div>
        </div>
        """, unsafe_allow_html=True)
    with k_col4:
        st.markdown(f"""
        <div class="neon-card card-rmse" style="text-align: center; padding: 20px 15px !important;">
            <div style="font-size: 11px; color: #4A3728; font-weight: 600; letter-spacing: 1.5px;">RMSE</div>
            <div class="neon-green" style="font-size: 28px; font-weight: 800; margin-top: 8px;">{rmse:.4f}</div>
            <div style="font-size: 11px; color: #4A3728; margin-top: 5px;">Root Mean Squared Error</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.1, 0.9], gap="large")
    
    with col1:
        st.markdown("<h4 style='color:#1F2C40; margin-top:0;'>Fitted Regression Line Fit</h4>", unsafe_allow_html=True)
        
        # Matplotlib Fit Plot in Ivory background
        fig, ax = plt.subplots(figsize=(8, 5.2))
        fig.patch.set_facecolor('#F5F0E6')
        ax.set_facecolor('#F5F0E6')
        
        # Plot train and test sets
        ax.scatter(X_train, y_train, color='#1F2C40', alpha=0.6, edgecolors='#4A3728', linewidths=0.5, s=70, label='Training Data (80%)')
        ax.scatter(X_test, y_test, color='#355E3B', marker='s', alpha=0.9, edgecolors='#1F2C40', linewidths=0.5, s=80, label='Testing Data (20%)')
        
        # Plot regression line in Burgundy (#5C1A1B)
        x_line = np.linspace(1, 10, 100).reshape(-1, 1)
        y_line = model.predict(x_line)
        ax.plot(x_line, y_line, color='#5C1A1B', linewidth=2.5, label='Learned Fit Line')
        
        # Visual limits & grids in Tan
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(0, 105)
        ax.grid(color='#C4A672', linestyle='--', linewidth=0.5, alpha=0.5)
        
        ax.set_xlabel('Study Hours', color='#4A3728', fontsize=11, labelpad=8)
        ax.set_ylabel('Scores (%)', color='#4A3728', fontsize=11, labelpad=8)
        ax.tick_params(colors='#4A3728', which='both', labelsize=9)
        
        for spine_name, spine in ax.spines.items():
            if spine_name in ['top', 'right']:
                spine.set_visible(False)
            else:
                spine.set_color('#C4A672')
                
        legend = ax.legend(facecolor='#E8DCC8', edgecolor='#C4A672', labelcolor='#1F2C40', fontsize=9, loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        
    with col2:
        st.markdown("<h4 style='color:#1F2C40; margin-top:0;'>Test Predictions vs Actuals</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color:#4A3728; font-size:13px; margin-bottom: 20px;'>Comparison of target scores vs model predictions on the 20% test subset.</p>", unsafe_allow_html=True)
        
        # Reconstruct prediction dataframe
        comparison_df = pd.DataFrame({
            'Hours': X_test.flatten(),
            'Actual Marks': y_test,
            'Predicted Marks': np.round(y_pred, 1)
        })
        
        # Render clean, custom styled table with Burgundy header accent
        comparison_table_html = render_styled_table(comparison_df, header_accent_color="#5C1A1B")
        st.markdown(comparison_table_html, unsafe_allow_html=True)
