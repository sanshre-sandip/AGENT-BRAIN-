import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from langchain_chroma import Chroma
from .embedding import get_embedder
from .config import CHROMA_DIR

# ----------------------------------
# Setup
# ----------------------------------
router = APIRouter()
logger = logging.getLogger(__name__)

# CHROMA_DIR is now imported from .config

_vectorstore = None
...
def get_vectorstore():
    """
    Lazy-load the Chroma vector store.
    """
    global _vectorstore
    if _vectorstore is None:
        embedder = get_embedder()
        logger.info(f"Connecting to Chroma DB — persist dir: {CHROMA_DIR}")
        _vectorstore = Chroma(
            collection_name="web_bot",
            embedding_function=embedder,
            persist_directory=CHROMA_DIR,
        )
    return _vectorstore
...

# ----------------------------------
# Request / Response Models
# ----------------------------------
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


# ----------------------------------
# Endpoints
# ----------------------------------

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
        store = get_vectorstore()
        
        # Using get() to check if empty as a workaround for missing public count()
        # We only need to know if it's 0, so we limit to 1
        existing = store.get(limit=1)
        if not existing["ids"]:
            raise HTTPException(status_code=404, detail="Vector store is empty — store some chunks first")

        results = store.similarity_search_with_relevance_scores(
            query=request.query,
            k=request.top_k,
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