#!/bin/bash
# setup.sh
# One command to install everything and run the full project.
#
# Usage:
#   bash setup.sh

echo "================================================="
echo "  Personal Expense Predictor — Setup & Run"
echo "================================================="

echo ""
echo "[1/2] Installing Python dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "[2/2] Running the ML pipeline..."
python main.py
