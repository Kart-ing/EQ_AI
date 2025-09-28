#!/bin/bash
# Start the React Frontend

# Configuration
FRONTEND_DIR="/home/memryx/Desktop/EQ_AI/audio-mastering-frontend"
API_URL="http://localhost:8000"

echo "Starting React Audio Mastering Frontend..."
echo "Frontend Directory: $FRONTEND_DIR"
echo "API Server: $API_URL"

# Check if frontend directory exists
if [ ! -d "$FRONTEND_DIR" ]; then
    echo "Error: Frontend directory not found at $FRONTEND_DIR"
    echo "Please run the React setup script first"
    exit 1
fi

# Navigate to frontend directory
cd "$FRONTEND_DIR" || exit 1
echo "Changed to directory: $(pwd)"

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "Error: package.json not found. This doesn't appear to be a React project."
    exit 1
fi

# Check if node_modules exists, install if not
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

# Check if API server is running
echo "Checking if API server is running..."
if curl -s "$API_URL/health" > /dev/null 2>&1; then
    echo "✅ API server is running at $API_URL"
else
    echo "⚠️  API server not detected at $API_URL"
    echo "Please start the API server first with:"
    echo "  ./start_api_server.sh"
    echo ""
    echo "Continuing anyway - you can start the API server later..."
fi

echo ""
echo "Starting React development server..."
echo "Frontend will be available at: http://localhost:3000"
echo "Press Ctrl+C to stop"
echo ""

# Start the React development server
npm run dev