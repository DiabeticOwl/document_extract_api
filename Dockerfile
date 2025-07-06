# --- Stage 1: Get the Ollama Binary ---
# Use a multi-stage build to securely and reliably get the Ollama binary
# from the official image without installing its entire environment.
FROM ollama/ollama:latest AS ollama

# --- Stage 2: Build Our Application ---
# Start from a secure, minimal, and up-to-date Python base image.
FROM python:3.13.5-slim-bookworm

# --- Environment Variables ---
# This section sets up the container's environment. These variables are crucial
# for ensuring that all tools and libraries write their data to user-owned,
# predictable locations, which is the key to avoiding permission errors.

# Sets the home directory for the current user. Many tools (like pip, EasyOCR)
# default to writing cache and config files in the user's home (`~`).
# By setting this explicitly, we control where these files are written.
ENV HOME=/home/user

# This is a specific variable for Ollama. It tells the Ollama server exactly
# where to download and store its large model files. We place it inside
# the user's home directory to guarantee write permissions.
ENV OLLAMA_MODELS=${HOME}/.ollama/models

# This variable follows the XDG Base Directory Specification. Libraries like
# `huggingface_hub` (used by sentence-transformers) respect this and will
# store their cache here, preventing permission errors.
ENV XDG_CACHE_HOME=${HOME}/.cache

# This tells the Python interpreter not to buffer stdout and stderr. In a Docker
# environment, this is essential for viewing logs in real-time.
ENV PYTHONUNBUFFERED=1

# Adds the user's local binary directory to the system's PATH. When pip
# installs packages for a user, it places executables (like uvicorn) here.
# This ensures the shell can find and run them.
ENV PATH="${HOME}/.local/bin:${PATH}"


# --- Root-level Setup ---
# All commands in this section run as the root user.
# Create a standard, non-root user and its home directory for better security.
RUN groupadd -r user -g 1000 && \
    useradd -r -u 1000 -g user -d ${HOME} -m -s /bin/bash user

# Install system dependencies needed for the application.
RUN apt-get update && \
    apt-get upgrade -y --no-install-recommends && \
    apt-get install git wget curl -y && \
    rm -rf /var/lib/apt/lists/*

# Proactively create all directories the application will need to write to as root.
RUN mkdir -p ${OLLAMA_MODELS} ${XDG_CACHE_HOME} && \
    touch ${HOME}/.gitconfig && \
    # Recursively change the ownership of the entire home directory to our user.
    chown -R user:user ${HOME}

# --- User-level Execution ---
# Switch to the non-root user for all subsequent commands.
USER user
WORKDIR ${HOME}

# Copy the Ollama binary from the first stage into a system-wide path.
COPY --from=ollama /bin/ollama /usr/local/bin/

# Copy application files and ensure the user owns them.
COPY --chown=user:user . .

# Install Python dependencies into the user's local site-packages.
RUN pip install --no-cache-dir -r requirements.txt

# Make the startup script executable.
RUN chmod +x ./start.sh

# Expose the port the app runs on.
EXPOSE 7860

# Define the command to run when the container starts.
CMD ["./start.sh"]
