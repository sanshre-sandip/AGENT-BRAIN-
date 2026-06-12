import logging
from fastapi import FastAPI
import uvicorn
import document_loader
import chunking
import embedding
import vectorstore
import db
import retriever



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Web Bot Backend",
    description="Load data from websites or PDFs",
    version="1.0.0"
)

app.include_router(document_loader.router, prefix="/process")
app.include_router(chunking.router, prefix="/chunk") 
app.include_router(embedding.router, prefix="/embed")    
app.include_router(vectorstore.router, prefix="/vectorstore")      
app.include_router(db.router, prefix="/db")     
app.include_router(retriever.router, prefix="/retrieve")

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Web Bot Backend"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)