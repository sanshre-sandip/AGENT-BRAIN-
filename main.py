import logging
from fastapi import FastAPI
import uvicorn
import document_loader
import chunking

# ----------------------------------
# Logging
# ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------------
# FastAPI
# ----------------------------------
app = FastAPI(
    title="Web Bot Backend",
    description="Load data from websites or PDFs",
    version="1.0.0"
)

# Include the router from document_loader
app.include_router(document_loader.router, prefix="/process")
app.include_router(chunking.router, prefix="/chunk")

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Web Bot Backend"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
