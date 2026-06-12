from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
import os

app = FastAPI(title="Web Bot Backend")

class SourceRequest(BaseModel):
    source: str
    type: str = "web"  # "web" or "pdf"

@app.get("/")
async def root():
    return {"message": "Web Bot Backend is running"}

@app.post("/process")
async def process_source(request: SourceRequest):
    if not request.source:
        raise HTTPException(status_code=400, detail="Source cannot be empty")
    
    try:
        if request.type == "pdf":
            if not os.path.exists(request.source):
                raise HTTPException(status_code=404, detail=f"PDF file not found: {request.source}")
            loader = PyPDFLoader(request.source)
            docs = loader.load()
        else:
            # Default to web
            loader = WebBaseLoader(request.source)
            docs = loader.load()
        
        # Process loaded documents
        content = "\n".join([doc.page_content for doc in docs])
        
        result = {
            "status": "success",
            "loader_used": request.type,
            "document_count": len(docs),
            "processed_length": len(content),
            "preview": content[:500] + "..." if len(content) > 500 else content,
            "message": f"Successfully loaded data using {request.type} loader"
        }
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
