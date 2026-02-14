#!/bin/bash

# ASL Translation System - Simple Runner
cd "$(dirname "$0")"

echo "🚀 Starting ASL Translation System (Camera Test)..."
echo "Using Python environment: .venv39"

# Run with the correct virtual environment
./.venv39/bin/streamlit run app_simple.py

echo "✅ App stopped."
