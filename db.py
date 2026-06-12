import logging
from collections import Counter

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from vectorstore import vectorstore

router = APIRouter()

logger = logging.getLogger(__name__)


class SourceRequest(BaseModel):
    source: str


@router.get("/info")
async def db_info():
    """
    Database overview.
    """
    try:
        result = vectorstore._collection.get(
            include=["documents", "metadatas"]
        )

        total_docs = len(result["ids"])

        sources = [
            meta.get("source", "unknown")
            for meta in result["metadatas"]
        ]

        source_counts = dict(Counter(sources))

        return {
            "total_documents": total_docs,
            "unique_sources": len(source_counts),
            "sources": source_counts,
            "persist_directory": "./chroma_db",
            "embedding_model": "all-MiniLM-L6-v2"
        }

    except Exception as e:
        logger.exception("DB info failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources")
async def list_sources():
    """
    List all sources and chunk counts.
    """
    try:
        result = vectorstore._collection.get(
            include=["metadatas"]
        )

        sources = [
            meta.get("source", "unknown")
            for meta in result["metadatas"]
        ]

        counts = Counter(sources)

        output = [
            {
                "source": source,
                "chunk_count": count
            }
            for source, count in counts.items()
        ]

        output.sort(
            key=lambda x: x["chunk_count"],
            reverse=True
        )

        return {
            "total_sources": len(output),
            "sources": output
        }

    except Exception as e:
        logger.exception("List sources failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_chunks(
    source: str | None = None,
    limit: int = 20,
    offset: int = 0
):
    """
    Paginated chunk listing.
    """
    try:
        result = vectorstore._collection.get(
            include=["documents", "metadatas"]
        )

        ids = result["ids"]
        docs = result["documents"]
        metas = result["metadatas"]

        chunks = []

        for chunk_id, doc, meta in zip(ids, docs, metas):

            chunk_source = meta.get(
                "source",
                "unknown"
            )

            if source and chunk_source != source:
                continue

            chunks.append({
                "id": chunk_id,
                "source": chunk_source,
                "char_count": len(doc),
                "preview": doc[:200]
            })

        total_matching = len(chunks)

        paginated = chunks[offset:offset + limit]

        return {
            "source_filter": source,
            "total_matching": total_matching,
            "returned": len(paginated),
            "limit": limit,
            "offset": offset,
            "chunks": paginated
        }

    except Exception as e:
        logger.exception("List chunks failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get/{chunk_id}")
async def get_chunk(chunk_id: str):
    """
    Fetch one chunk by ID.
    """
    try:
        result = vectorstore._collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"]
        )

        ids = result.get("ids", [])

        if not ids:
            raise HTTPException(
                status_code=404,
                detail=f"Chunk not found: {chunk_id}"
            )

        return {
            "status": "success",
            "id": ids[0],
            "source": result["metadatas"][0].get(
                "source",
                "unknown"
            ),
            "text": result["documents"][0]
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Get chunk failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear-source")
async def clear_source(request: SourceRequest):
    """
    Delete all chunks from a source.
    """
    try:
        source = request.source.strip()

        if not source:
            raise HTTPException(
                status_code=400,
                detail="Source cannot be empty"
            )

        result = vectorstore._collection.get(
            include=["metadatas"]
        )

        ids_to_delete = []

        for chunk_id, meta in zip(
            result["ids"],
            result["metadatas"]
        ):
            if meta.get("source") == source:
                ids_to_delete.append(chunk_id)

        if not ids_to_delete:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for source: {source}"
            )

        vectorstore._collection.delete(
            ids=ids_to_delete
        )

        remaining = vectorstore._collection.count()

        return {
            "message": f"Removed source '{source}'",
            "chunks_deleted": len(ids_to_delete),
            "total_remaining": remaining
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Clear source failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear-all")
async def clear_all():
    """
    Delete entire database.
    """
    try:
        count_before = vectorstore._collection.count()

        result = vectorstore._collection.get()

        if result["ids"]:
            vectorstore._collection.delete(
                ids=result["ids"]
            )

        return {
            "message": "Database cleared",
            "chunks_deleted": count_before,
            "total_remaining": 0
        }

    except Exception as e:
        logger.exception("Clear all failed")
        raise HTTPException(status_code=500, detail=str(e))