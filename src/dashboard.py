"""
dashboard.py
------------
A premium AMOLED-themed dashboard for the Student Marks Prediction project.
Features a pitch-black theme, interactive floating background particles,
glowing card layouts, and dynamic predictions using the trained Linear Regression model.

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
    page_title="Student Marks Predictor - AMOLED Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. AMOLED and Animated Background CSS Injections
custom_css = """
<style>
/* Reset and AMOLED pitch black background */
.stApp {
    background-color: #000000 !important;
    color: #ffffff !important;
}

/* Hide Streamlit elements for clean app feel */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom keyframes for floating bokeh bubbles background */
@keyframes floatUp {
    0% { transform: translateY(0px) rotate(0deg); opacity: 0; }
    10% { opacity: 0.3; }
    90% { opacity: 0.3; }
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
    background-color: rgba(0, 243, 255, 0.08);
    bottom: -150px;
    animation: floatUp 25s infinite linear;
    border-radius: 50%;
    box-shadow: 0 0 15px rgba(0, 243, 255, 0.15);
}

/* Custom positioning, sizing and coloring for the floating particles */
.bg-bubbles li:nth-child(1) { left: 8%; width: 75px; height: 75px; animation-delay: 0s; animation-duration: 20s; background-color: rgba(255, 0, 127, 0.06); box-shadow: 0 0 25px rgba(255, 0, 127, 0.12); }
.bg-bubbles li:nth-child(2) { left: 18%; width: 30px; height: 30px; animation-delay: 3s; animation-duration: 23s; }
.bg-bubbles li:nth-child(3) { left: 28%; width: 45px; height: 45px; animation-delay: 6s; animation-duration: 17s; }
.bg-bubbles li:nth-child(4) { left: 45%; width: 60px; height: 60px; animation-delay: 1s; animation-duration: 27s; background-color: rgba(255, 0, 127, 0.06); box-shadow: 0 0 25px rgba(255, 0, 127, 0.12); }
.bg-bubbles li:nth-child(5) { left: 58%; width: 38px; height: 38px; animation-delay: 4s; animation-duration: 20s; }
.bg-bubbles li:nth-child(6) { left: 72%; width: 100px; height: 100px; animation-delay: 8s; animation-duration: 32s; }
.bg-bubbles li:nth-child(7) { left: 85%; width: 50px; height: 50px; animation-delay: 2s; animation-duration: 25s; background-color: rgba(0, 243, 255, 0.06); }
.bg-bubbles li:nth-child(8) { left: 93%; width: 25px; height: 25px; animation-delay: 11s; animation-duration: 19s; }
.bg-bubbles li:nth-child(9) { left: 38%; width: 85px; height: 85px; animation-delay: 14s; animation-duration: 30s; background-color: rgba(255, 0, 127, 0.06); box-shadow: 0 0 25px rgba(255, 0, 127, 0.12); }
.bg-bubbles li:nth-child(10) { left: 65%; width: 55px; height: 55px; animation-delay: 5s; animation-duration: 21s; }

/* Custom glowing card panels */
.neon-card {
    background-color: #050505 !important;
    border: 1px solid #1c1c1e !important;
    border-radius: 15px !important;
    padding: 22px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8) !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}

.neon-card:hover {
    border-color: #00f3ff !important;
    box-shadow: 0 0 15px rgba(0, 243, 255, 0.2) !important;
    transform: translateY(-4px);
}

.neon-card-pink:hover {
    border-color: #ff007f !important;
    box-shadow: 0 0 15px rgba(255, 0, 127, 0.2) !important;
}

/* Interactive slider elements and labels */
.stSlider label {
    color: #a5a5a5 !important;
    font-size: 14px !important;
}

/* Neon text styles */
.glow-title {
    color: #ffffff;
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 5px;
    letter-spacing: 1px;
    text-shadow: 0 0 12px rgba(255, 255, 255, 0.1);
}

.glow-subtitle {
    color: #888888;
    font-size: 16px;
    text-align: center;
    margin-bottom: 30px;
}

.neon-blue {
    color: #00f3ff;
    text-shadow: 0 0 8px rgba(0, 243, 255, 0.4);
    font-weight: bold;
}

.neon-pink {
    color: #ff007f;
    text-shadow: 0 0 8px rgba(255, 0, 127, 0.4);
    font-weight: bold;
}

.neon-green {
    color: #39ff14;
    text-shadow: 0 0 12px rgba(57, 255, 20, 0.6);
    font-weight: 800;
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

# Render styling and animated background
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
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

# Slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

# 4. App Header UI
st.markdown("<div class='glow-title'>🎓 Student Marks Prediction Studio</div>", unsafe_allow_html=True)
st.markdown("<div class='glow-subtitle'>Interactive AMOLED Dashboard powered by Linear Regression</div>", unsafe_allow_html=True)

# 5. Grid Layout
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("<h3 style='color: #ffffff; border-bottom: 1px solid #1c1c1e; padding-bottom: 8px;'>Interactive Predictor</h3>", unsafe_allow_html=True)
    
    st.write("")
    
    # Study hours slider widget
    study_hours = st.slider(
        "Slide to choose Daily Study Hours:",
        min_value=0.0,
        max_value=12.0,
        value=6.5,
        step=0.25
    )
    
    # Calculate Prediction
    predicted_score = model.predict([[study_hours]])[0]
    predicted_score_clipped = np.clip(predicted_score, 0, 100)
    
    # Prediction Display Card
    st.markdown(f"""
    <div class="neon-card" style="text-align: center; margin-top: 25px;">
        <h4 style="margin: 0; color: #888888; font-weight: 500; font-size: 16px;">ESTIMATED SCORE</h4>
        <div class="neon-green" style="font-size: 56px; margin: 10px 0;">{predicted_score_clipped:.1f}%</div>
        <p style="margin: 0; color: #a5a5a5; font-size: 13px;">
            For studying <span class="neon-blue">{study_hours:.2f} hours</span> per day
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Equation Display Card
    st.markdown(f"""
    <div class="neon-card neon-card-pink" style="text-align: center;">
        <h5 style="margin: 0; color: #888888; font-weight: 500; font-size: 14px; margin-bottom: 8px;">LEARNED REGRESSION MODEL</h5>
        <code style="background-color: transparent; color: #ff007f; font-size: 15px; font-weight: bold; font-family: monospace;">
            Marks = {slope:.3f} * Hours + {intercept:.3f}
        </code>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics Column Cards
    st.markdown("<h4 style='color: #ffffff; margin-top: 25px; margin-bottom: 15px; font-size: 16px;'>Model Performance Metrics</h4>", unsafe_allow_html=True)
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(f"""
        <div class="neon-card" style="text-align: center; padding: 12px !important; margin-bottom: 0px !important;">
            <div style="font-size: 11px; color: #888888; font-weight: 500;">R² SCORE</div>
            <div class="neon-blue" style="font-size: 20px; font-weight: bold; margin-top: 5px;">{r2:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="neon-card" style="text-align: center; padding: 12px !important; margin-bottom: 0px !important;">
            <div style="font-size: 11px; color: #888888; font-weight: 500;">MAE</div>
            <div class="neon-pink" style="font-size: 20px; font-weight: bold; margin-top: 5px;">{mae:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="neon-card" style="text-align: center; padding: 12px !important; margin-bottom: 0px !important;">
            <div style="font-size: 11px; color: #888888; font-weight: 500;">RMSE</div>
            <div class="neon-green" style="font-size: 20px; font-weight: bold; margin-top: 5px;">{rmse:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("<h3 style='color: #ffffff; border-bottom: 1px solid #1c1c1e; padding-bottom: 8px;'>Regression Visualization</h3>", unsafe_allow_html=True)
    
    # Render Matplotlib Figure styled in pitch-black AMOLED mode
    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#000000')
    
    # Plot dataset
    ax.scatter(df['Hours'], df['Scores'], color='#00f3ff', alpha=0.8, edgecolors='#ffffff', linewidths=0.5, s=80, label='Student Data')
    
    # Plot current slider coordinate as a glowing point
    ax.scatter([study_hours], [predicted_score_clipped], color='#39ff14', s=160, edgecolors='#ffffff', linewidths=1.5, zorder=5, label='Your Prediction')
    
    # Plot fit line
    x_line = np.linspace(1, 10, 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax.plot(x_line, y_line, color='#ff007f', linewidth=2.5, label=f'Model Fit (R² = {r2:.2f})')
    
    # Visual grid and axes
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 105)
    ax.grid(color='#181818', linestyle='--', linewidth=0.7)
    
    # Custom Labels
    ax.set_xlabel('Study Hours', color='#888888', fontsize=11, labelpad=8)
    ax.set_ylabel('Scores (%)', color='#888888', fontsize=11, labelpad=8)
    ax.tick_params(colors='#888888', which='both', labelsize=9)
    
    # Hide outer boundaries (spines) except bottom/left to keep it clean
    for spine_name, spine in ax.spines.items():
        if spine_name in ['top', 'right']:
            spine.set_visible(False)
        else:
            spine.set_color('#222222')
            
    # Legend
    legend = ax.legend(facecolor='#050505', edgecolor='#1c1c1e', labelcolor='#ffffff', fontsize=9, loc='upper left')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Info Panel
    st.markdown("""
    <div class="neon-card" style="margin-top: 15px; font-size: 13px; line-height: 1.5; color: #a5a5a5;">
        💡 <b>Supervised Learning Concept</b>: 
        The blue dots represent the 25 historical records (study hours, actual score). 
        The pink line represents the trained Linear Regression model. 
        Adjusting the slider updates the green point <span class="neon-green">Your Prediction</span> 
        live along this line, illustrating how prediction values map exactly onto the learned model equation.
    </div>
    """, unsafe_allow_html=True)
