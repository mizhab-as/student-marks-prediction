#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Clear the screen
clear

echo "=============================================================="
echo "    🎓 Student Marks Prediction - Project Launcher 🎓"
echo "=============================================================="
echo ""

# Resolve absolute path to project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 1. Dependency Verification
echo "[*] Step 1/3: Installing & Verifying Dependencies..."
pip3 install pandas numpy matplotlib scikit-learn streamlit
echo "[✓] Dependencies verified."
echo ""

# 2. Dataset Preparation
echo "=============================================================="
echo "[*] Step 2/3: Preparing Student Scores Dataset..."
echo "=============================================================="
python3 src/generate_dataset.py
echo ""

# 3. Model Training & Evaluation Pipeline
echo "=============================================================="
echo "[*] Step 3/3: Running Regression Training & ML Pipeline..."
echo "=============================================================="
python3 src/marks_prediction.py
echo ""

# 4. Interactive Dashboard Launch Prompt
echo "=============================================================="
echo "[?] Dashboard Launch Launch Options"
echo "=============================================================="
read -p "Would you like to start the Streamlit web dashboard? (y/n): " launch_dashboard
echo ""

if [[ "$launch_dashboard" =~ ^[Yy]$ ]]; then
    echo "[*] Starting Streamlit dashboard..."
    echo "[i] Press Ctrl+C in this terminal window to stop the dashboard server when finished."
    streamlit run src/dashboard.py
else
    echo "[✓] Launch skipped. You can manually launch the dashboard at any time with:"
    echo "    streamlit run src/dashboard.py"
fi
echo ""
echo "[✓] Execution completed successfully!"
