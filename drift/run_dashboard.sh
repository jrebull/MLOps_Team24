#!/bin/bash

echo ""
echo "============================================"
echo "🔥 Data Drift Detection Dashboard"
echo "MLOps Team 24"
echo "============================================"
echo ""

echo "📦 Checking dependencies..."
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "⚙️  Installing Streamlit and Plotly..."
    pip3 install streamlit==1.28.1 plotly==5.17.0 pandas==2.0.3 numpy==1.24.3
    echo "✅ Dependencies installed"
else
    echo "✅ Streamlit already installed"
fi

echo ""
echo "============================================"
echo "🚀 Starting Dashboard..."
echo "============================================"
echo ""
echo "📱 Access at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop"
echo ""

streamlit run drift_streamlit_dashboard.py --logger.level=error