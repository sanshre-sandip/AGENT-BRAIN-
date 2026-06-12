import os

# Centralized RAG Configuration
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BACKEND_URL = os.getenv("BACKEND", "http://localhost:8000")
