#!/bin/bash
# Quick start script for ConformAI demo

echo "🚀 Starting ConformAI Agent Demo..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing Streamlit..."
    uv pip install streamlit
fi

echo "✅ Dependencies ready"
echo ""
echo "🎨 Launching interactive UI..."
echo "   → Opening http://localhost:8501"
echo ""
echo "💡 Try these queries:"
echo "   • Simple: 'What is a high-risk AI system?'"
echo "   • Medium: 'What are the documentation requirements for high-risk AI systems?'"
echo "   • Complex: 'Compare recruitment AI vs healthcare AI documentation requirements'"
echo ""
echo "🛑 Press Ctrl+C to stop"
echo ""

streamlit run app_demo.py
