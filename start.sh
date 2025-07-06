#!/bin/bash

# Start the Ollama server in the background.
# The `ollama` command is found in /app/bin, which we added to the PATH.
ollama serve &

# Wait a moment for the server to start up.
sleep 5

# Pull the phi3:mini model. This will be stored in the /app/ollama_models
# directory as configured by the OLLAMA_MODELS environment variable.
echo "Pulling phi3:mini model..."
ollama pull phi3:mini

# This command forces Ollama to load the phi3:mini model into GPU memory.
# It sends a trivial prompt and waits for the response. By doing this now,
# we ensure the model is "hot and ready" before the API server starts
# accepting requests, which prevents the first user request from timing out.
echo "Warming up the phi3:mini model..."
ollama run phi3:mini "Hello! Please respond with just 'OK' to warm up."

echo "Starting FastAPI server..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 7860
