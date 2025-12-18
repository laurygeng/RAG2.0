#!/bin/bash

# Stop on error
set -e

echo "🚀 Starting RAG System Setup..."

# 1. Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 could not be found. Please install Python 3."
    exit 1
fi

echo "✅ Python 3 found."

# 2. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment 'venv'..."
    python3 -m venv venv
else
    echo "ℹ️  Virtual environment 'venv' already exists."
fi

# 3. Activate Virtual Environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# 4. Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# 5. Install Dependencies
if [ -f "requirements.txt" ]; then
    echo "📥 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found! Please ensure it is in the same directory."
    exit 1
fi

# 6. Download NLTK Data
echo "📚 Downloading NLTK data (punkt, wordnet, omw-1.4)..."
python -m nltk.downloader punkt wordnet omw-1.4

# 7. Check for Ollama (Optional but recommended)
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama command not found. If you need to run generation, please ensure Ollama is installed and running."
    echo "   You can install it from https://ollama.com/"
else
    echo "✅ Ollama found."
fi

echo ""
echo "🎉 Setup Complete!"
echo ""
echo "To run the code, first activate the environment:"
echo "   source venv/bin/activate"
echo ""
echo "Then you can run the scripts in the 'code' directory. Examples:"
echo ""
echo "1. Generate Answers:"
echo "   python code/generate_answers.py"
echo ""
echo "2. Evaluate Answers (BLEU):"
echo "   python code/evaluate_answers_BLEU.py results/your_results_file.csv"
echo ""
echo "3. Evaluate Answers (BERTScore):"
echo "   python code/evaluate_answers_BERTSCORE.py results/your_results_file.csv"
echo ""
echo "4. Evaluate Answers (METEOR):"
echo "   python code/evaluate_answers_METEOR.py results/your_results_file.csv"
echo ""
echo "5. Evaluate Answers (F1):"
echo "   python code/evaluate_answers_F1.py results/your_results_file.csv"
echo ""
echo "6. Evaluate Answers (ROUGE):"
echo "   python code/evaluate_answers_ROUGE.py results/your_results_file.csv"
echo ""
