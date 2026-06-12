import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from .config import (
    LLM_PROVIDER, OPENAI_API_KEY, ANTHROPIC_API_KEY, 
    GOOGLE_API_KEY, OLLAMA_BASE_URL
)

# Optional imports based on provider
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_ollama import ChatOllama
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        ChatOllama = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

# Try to import retriever logic for unified RAG endpoint
try:
    from RAG.vectorstore import get_vectorstore
except ImportError:
    get_vectorstore = None

# Setup router
router = APIRouter()
logger = logging.getLogger(__name__)

# --- Models ---

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class GenerateRequest(BaseModel):
    query: str
    context: List[str]
    history: Optional[List[Message]] = None
    provider: Optional[str] = None  # openai, anthropic, ollama, google
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class RagRequest(BaseModel):
    query: str
    source: Optional[str] = None
    k: int = 5
    history: Optional[List[Message]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class GenerateResponse(BaseModel):
    status: str
    answer: str
    model_used: str
    provider_used: str

# --- Helpers ---

def get_llm(provider: str, model: Optional[str], temperature: float = 0.7, streaming: bool = False):
    """
    Factory function to get the LLM instance based on provider and model.
    """
    provider = provider.lower()
    
    if provider == "openai":
        if not ChatOpenAI:
            raise HTTPException(status_code=500, detail="langchain-openai not installed")
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not found in environment")
        return ChatOpenAI(model=model or "gpt-4o", api_key=OPENAI_API_KEY, temperature=temperature, streaming=streaming)
    
    elif provider == "anthropic":
        if not ChatAnthropic:
            raise HTTPException(status_code=500, detail="langchain-anthropic not installed")
        if not ANTHROPIC_API_KEY:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not found in environment")
        return ChatAnthropic(model=model or "claude-3-5-sonnet-20240620", api_key=ANTHROPIC_API_KEY, temperature=temperature, streaming=streaming)
    
    elif provider == "google":
        if not ChatGoogleGenerativeAI:
            raise HTTPException(status_code=500, detail="langchain-google-genai not installed")
        if not GOOGLE_API_KEY:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not found in environment")
        return ChatGoogleGenerativeAI(model=model or "gemini-1.5-pro", google_api_key=GOOGLE_API_KEY, temperature=temperature, streaming=streaming)
    
    elif provider == "ollama":
        if not ChatOllama:
            raise HTTPException(status_code=500, detail="langchain-ollama or langchain-community not installed")
        return ChatOllama(model=model or "llama3", base_url=OLLAMA_BASE_URL, temperature=temperature, streaming=streaming)
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

def construct_prompt(query: str, context: List[str], history: Optional[List[Message]] = None):
    """
    Constructs a prompt with context and history.
    """
    system_msg = (
        "You are a helpful AI assistant. Use the provided context to answer the user's question.\n"
        "If the answer is not contained within the context, politely state that you don't know based on the provided information.\n"
        "Keep your answer concise and well-structured."
    )
    
    context_text = "\n---\n".join(context)
    
    prompt = f"{system_msg}\n\nContext:\n{context_text}\n\n"
    
    if history:
        prompt += "Conversation History:\n"
        for msg in history:
            prompt += f"{msg.role.capitalize()}: {msg.content}\n"
        prompt += "\n"
    
    prompt += f"User Question: {query}\n"
    prompt += "Assistant Answer:"
    
    return prompt

# --- Endpoints ---

@router.post("", response_model=GenerateResponse)
async def generate_answer(request: GenerateRequest):
    """
    Generates an answer using the provided query and context.
    """
    provider = request.provider or LLM_PROVIDER
    
    try:
        llm = get_llm(provider, request.model, request.temperature, request.stream)
        prompt = construct_prompt(request.query, request.context, request.history)
        
        logger.info(f"Generating answer using {provider}...")
        
        if request.stream:
            async def stream_generator():
                async for chunk in llm.astream(prompt):
                    content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    yield content
            return StreamingResponse(stream_generator(), media_type="text/plain")
        
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return GenerateResponse(
            status="success",
            answer=answer,
            model_used=request.model or "default",
            provider_used=provider
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag", response_model=GenerateResponse)
async def rag_answer(request: RagRequest):
    """
    Unified RAG endpoint: Retrieves context and then generates an answer.
    """
    if get_vectorstore is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized or RAG modules not found")
    
    provider = request.provider or LLM_PROVIDER
    
    try:
        # 1. Retrieve context
        logger.info(f"Retrieving context for query: {request.query}")
        store = get_vectorstore()
        filter_dict = {"source": request.source} if request.source else None
        docs = store.similarity_search(request.query, k=request.k, filter=filter_dict)
        context = [doc.page_content for doc in docs]
        
        if not context:
            logger.warning("No relevant context found in vector store.")
            context = ["No relevant context found."]
            
        # 2. Generate answer
        llm = get_llm(provider, request.model, request.temperature, request.stream)
        prompt = construct_prompt(request.query, context, request.history)
        
        logger.info(f"Generating RAG answer using {provider}...")
        
        if request.stream:
            async def stream_generator():
                async for chunk in llm.astream(prompt):
                    content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    yield content
            return StreamingResponse(stream_generator(), media_type="text/plain")
        
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return GenerateResponse(
            status="success",
            answer=answer,
            model_used=request.model or "default",
            provider_used=provider
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("RAG Generation failed")
        raise HTTPException(status_code=500, detail=str(e))
