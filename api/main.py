from fastapi import FastAPI
from api import endpoints

app = FastAPI(
    title="Intelligent Document Understanding API",
    description="An API that extracts structured information from documents using OCR and AI.",
    version="1.0.0"
)

# Include the router from the endpoints module.
# This makes the API modular by keeping the endpoint logic in a separate file.
# All routes defined in the 'endpoints.py' file will now be part of the main app.
app.include_router(endpoints.router)

# A simple root endpoint to confirm the API is running.
@app.get("/", tags=["Status"])
def read_root():
    return {"status": "API is running"}
