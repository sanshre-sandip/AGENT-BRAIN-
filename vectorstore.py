import logging
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------------
# Setup
# ----------------------------------
router = APIRouter()
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"  # local folder where Chroma persists data

# Load embedder once at startup
logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
logger.info("Embedding model ready")

# Connect to persistent Chroma DB
vectorstore = Chroma(
    collection_name="web_bot",
    embedding_function=embedder,
    persist_directory=CHROMA_DIR,
)
logger.info(f"Chroma DB connected — persist dir: {CHROMA_DIR}")


# ----------------------------------
# Request / Response Models
# ----------------------------------
class StoreRequest(BaseModel):
    chunks: list[str]               # Text chunks to store
    source: Optional[str] = None    # Optional source label (URL or file path)


class StoreResponse(BaseModel):
    status: str
    source: Optional[str]
    chunks_stored: int
    total_in_db: int
    ids: list[str]


class SearchRequest(BaseModel):
    query: str                      # Natural language query
    top_k: Optional[int] = 5       # Number of results to return


class SearchResult(BaseModel):
    rank: int
    score: float
    text: str
    source: Optional[str]


class SearchResponse(BaseModel):
    status: str
    query: str
    top_k: int
    results: list[SearchResult]


class DeleteRequest(BaseModel):
    source: str                     # Delete all chunks from this source


class StatsResponse(BaseModel):
    status: str
    total_documents: int
    persist_directory: str
    embedding_model: str


# ----------------------------------
# Endpoints
# ----------------------------------

@router.post("/store", response_model=StoreResponse)
async def store_chunks(request: StoreRequest):
    """
    Store text chunks into Chroma vector store.
    Each chunk is embedded and saved with a unique ID and source metadata.
    Data persists to disk at ./chroma_db
    """
    if not request.chunks:
        raise HTTPException(status_code=400, detail="Chunks list cannot be empty")

    valid_chunks = [c for c in request.chunks if c.strip()]
    if not valid_chunks:
        raise HTTPException(status_code=400, detail="All chunks are empty or whitespace")

    try:
        # Generate unique IDs for each chunk
        ids = [str(uuid.uuid4()) for _ in valid_chunks]

        # Attach source metadata to each chunk
        metadatas = [{"source": request.source or "unknown"} for _ in valid_chunks]

        vectorstore.add_texts(
            texts=valid_chunks,
            metadatas=metadatas,
            ids=ids,
        )

        total = vectorstore._collection.count()
        logger.info(f"Stored {len(valid_chunks)} chunks — total in DB: {total}")

        return StoreResponse(
            status="success",
            source=request.source,
            chunks_stored=len(valid_chunks),
            total_in_db=total,
            ids=ids,
        )

    except Exception as e:
        logger.exception("Store failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def search_chunks(request: SearchRequest):
    """
    Search the vector store using a natural language query.
    Returns top_k most semantically similar chunks with scores.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if request.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be a positive integer")

    try:
        total = vectorstore._collection.count()
        if total == 0:
            raise HTTPException(status_code=404, detail="Vector store is empty — store some chunks first")

        results = vectorstore.similarity_search_with_relevance_scores(
            query=request.query,
            k=min(request.top_k, total),
        )

        search_results = [
            SearchResult(
                rank=idx + 1,
                score=round(score, 4),
                text=doc.page_content,
                source=doc.metadata.get("source"),
            )
            for idx, (doc, score) in enumerate(results)
        ]

        logger.info(f"Search query={request.query!r} — returned {len(search_results)} results")

        return SearchResponse(
            status="success",
            query=request.query,
            top_k=request.top_k,
            results=search_results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete", response_model=dict)
async def delete_by_source(request: DeleteRequest):
    """
    Delete all chunks that belong to a specific source.
    """
    if not request.source.strip():
        raise HTTPException(status_code=400, detail="Source cannot be empty")

    try:
        # Fetch IDs matching the source
        results = vectorstore._collection.get(
            where={"source": request.source}
        )
        ids_to_delete = results.get("ids", [])

        if not ids_to_delete:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for source: {request.source!r}"
            )

        vectorstore._collection.delete(ids=ids_to_delete)
        total = vectorstore._collection.count()

        logger.info(f"Deleted {len(ids_to_delete)} chunks for source={request.source!r}")

        return {
            "status": "success",
            "source": request.source,
            "chunks_deleted": len(ids_to_delete),
            "total_in_db": total,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Delete failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Return stats about the current vector store.
    """
    try:
        total = vectorstore._collection.count()
        return StatsResponse(
            status="success",
            total_documents=total,
            persist_directory=CHROMA_DIR,
            embedding_model=EMBEDDING_MODEL,
        )
    except Exception as e:
        logger.exception("Stats failed")
        raise HTTPException(status_code=500, detail=str(e))