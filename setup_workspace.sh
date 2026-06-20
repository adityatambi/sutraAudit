#!/bin/bash
set -e

echo "Setting up Python virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing required libraries from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete! Run 'source venv/bin/activate' to use this environment."
