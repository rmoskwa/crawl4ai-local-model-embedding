FROM python:3.12-slim

ARG PORT=8051

WORKDIR /app

# Install uv
RUN pip install uv

# Copy the MCP server files
COPY . .

# Install packages directly to the system (no virtual environment)
# Combining commands to reduce Docker layers
RUN uv pip install --system -e . && \
    crawl4ai-setup

# Download BGE embedding model
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-large-en-v1.5', cache_dir='/app/models')"

# Set the BGE model path for the container
ENV BGE_MODEL_PATH=/app/models/models--BAAI--bge-large-en-v1.5/snapshots

EXPOSE ${PORT}

# Command to run the MCP server
CMD ["python", "src/crawl4ai_mcp.py"]
