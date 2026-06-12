import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

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
    from langchain_community.chat_models import ChatOllama
except ImportError:
    ChatOllama = None

# Setup router
router = APIRouter()
logger = logging.getLogger(__name__)

class GenerateRequest(BaseModel):
    query: str
    context: List[str]
    provider: Optional[str] = None  # openai, anthropic, ollama
    model: Optional[str] = None     # e.g., gpt-4o, claude-3-opus, llama3

class GenerateResponse(BaseModel):
    status: str
    answer: str
    model_used: str
    provider_used: str

def get_llm(provider: str, model: Optional[str]):
    """
    Factory function to get the LLM instance based on provider and model.
    """
    provider = provider.lower()
    
    if provider == "openai":
        if not ChatOpenAI:
            raise HTTPException(status_code=500, detail="langchain-openai not installed")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not found in environment")
        return ChatOpenAI(model=model or "gpt-4o", api_key=api_key)
    
    elif provider == "anthropic":
        if not ChatAnthropic:
            raise HTTPException(status_code=500, detail="langchain-anthropic not installed")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not found in environment")
        return ChatAnthropic(model=model or "claude-3-5-sonnet-20240620", api_key=api_key)
    
    elif provider == "ollama":
        if not ChatOllama:
            raise HTTPException(status_code=500, detail="langchain-community not installed")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model or "llama3", base_url=base_url)
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

@router.post("", response_model=GenerateResponse)
async def generate_answer(request: GenerateRequest):
    """
    Generates an answer using the provided query and context.
    The prompt is constructed as: Context + Query = Answer.
    """
    provider = request.provider or os.getenv("LLM_PROVIDER", "openai")
    
    try:
        llm = get_llm(provider, request.model)
        
        # Construct the Prompt
        context_text = "\n\n".join(request.context)
        prompt = (
            "You are a helpful assistant. Use the following context to answer the question.\n"
            "If the answer is not in the context, say that you don't know.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {request.query}\n\n"
            "Answer:"
        )
        
        logger.info(f"Generating answer using {provider}...")
        response = llm.invoke(prompt)
        
        # LangChain ChatModels return a BaseMessage, we want the content string
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
