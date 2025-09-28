#!/bin/bash
# Start the Audio Mastering API Server

# Configuration
PROJECT_DIR="/home/memryx/Desktop/EQ_AI/src"
VENV_NAME="mx"  # Your virtual environment name
GENRE_MODEL="/home/memryx/Desktop/EQ_AI/models/genre_model_50.dfp"
EQ_MODEL="/home/memryx/Desktop/EQ_AI/models/live_eq_model_50.dfp"

echo "Starting AI Audio Mastering API Server..."
echo "Project Directory: $PROJECT_DIR"
echo "Virtual Environment: $VENV_NAME"

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project directory not found at $PROJECT_DIR"
    exit 1
fi

# Navigate to project directory
cd "$PROJECT_DIR" || exit 1
echo "Changed to directory: $(pwd)"

# Activate virtual environment
if [ -f "/home/memryx/anaconda3/envs/$VENV_NAME/bin/activate" ]; then
    echo "Activating conda environment: $VENV_NAME"
    source "/home/memryx/anaconda3/envs/$VENV_NAME/bin/activate"
elif [ -f "$VENV_NAME/bin/activate" ]; then
    echo "Activating local virtual environment: $VENV_NAME"
    source "$VENV_NAME/bin/activate"
elif conda info --envs | grep -q "$VENV_NAME"; then
    echo "Activating conda environment: $VENV_NAME"
    conda activate "$VENV_NAME"
else
    echo "Warning: Virtual environment '$VENV_NAME' not found"
    echo "Proceeding with system Python..."
fi

# Check if models exist
if [ ! -f "$GENRE_MODEL" ]; then
    echo "Error: Genre model not found at $GENRE_MODEL"
    exit 1
fi

if [ ! -f "$EQ_MODEL" ]; then
    echo "Error: EQ model not found at $EQ_MODEL"
    exit 1
fi

# Check if API server file exists
if [ ! -f "server.py" ]; then
    echo "Error: audio_api_server.py not found in $(pwd)"
    echo "Available Python files:"
    ls -la *.py 2>/dev/null || echo "No Python files found"
    exit 1
fi

# Install dependencies if needed
echo "Installing/updating dependencies..."
pip install fastapi uvicorn python-multipart

echo ""
echo "Configuration:"
echo "Genre Model: $GENRE_MODEL"
echo "EQ Model: $EQ_MODEL"
echo "Server URL: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""

# Start the server
echo "Starting server..."
python server.py \
    --genre-model "$GENRE_MODEL" \
    --eq-model "$EQ_MODEL" \
    --host 0.0.0.0 \
    --port 8000