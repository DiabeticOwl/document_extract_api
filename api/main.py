from api import endpoints
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

# Create the main FastAPI application instance.
app = FastAPI(
    title="Intelligent Document Understanding API",
    description="An API that extracts structured information from documents using OCR and AI.",
    version="1.0.0"
)

# --- CORS Middleware ---
# This is the key to allowing the frontend to communicate with the API.
# It adds the necessary headers to the API's responses to tell browsers
# that it's safe to allow requests from other origins.

# Define the origins that are allowed to make requests.
# Using a wildcard "*" would allow all domains to make requests.
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://diabeticowl-document-extractor.hf.space",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

# Include the router from the endpoints module for our /extract_entities/ endpoint.
app.include_router(endpoints.router)

# Define the path to the frontend directory.
frontend_path = Path("frontend/index.html")

@app.get("/", response_class=FileResponse, tags=["Frontend"], include_in_schema=False)
async def read_index():
    """
    Serves the main HTML page for the web interface.
    This endpoint is configured to respond to GET requests at the root URL (`/`).
    When a user navigates to the base URL of the deployed application, this function
    is triggered. It uses FastAPI's `FileResponse` to directly return the
    contents of the `frontend/index.html` file, effectively serving the
    single-page application to the user's browser.
    """
    return frontend_path


@app.get("/health", tags=["Status"])
async def health_check():
    """
    Provides a simple, lightweight health check endpoint.
    This is crucial for production deployments. Platforms like Hugging Face Spaces
    or Kubernetes periodically send requests to a health check endpoint to verify
    that the application is running and responsive. By providing a fast, simple
    endpoint that doesn't trigger any heavy computation (like the AI pipeline),
    we prevent the platform from incorrectly assuming the application is down
    and restarting it during long-running tasks like model warm-ups.
    """
    return {"status": "ok"}
