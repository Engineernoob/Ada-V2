#!/bin/bash
echo "🚀 Setting up Ada v2.0 environment..."

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Create folders
mkdir -p core rl logs storage/models api voice
touch core/__init__.py rl/__init__.py api/__init__.py voice/__init__.py

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify
echo "✅ Environment ready!"
echo "Run Ada with: source .venv/bin/activate && python main.py"
