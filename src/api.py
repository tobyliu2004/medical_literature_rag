"""
FastAPI backend for Medical Literature RAG system.
Provides REST API endpoints for medical question answering.
| Endpoint        | What it means                  | Real-world analogy                  |
|-----------------|--------------------------------|-------------------------------------|
| GET /           | "Show me basic info"           | Like a restaurant's front door sign |
| GET /health     | "Are you working?"             | Like asking "Are you open?"         |
| POST /ask       | "Answer this medical question" | Like asking a doctor a question     |
| POST /search    | "Find papers about X"          | Like searching in a library         |
| GET /papers/123 | "Show me paper #123"           | Like asking for a specific book     |
| GET /stats      | "Show me your statistics"      | Like asking for a report            |
| GET /docs       | "Show me how to use you"       | Like reading an instruction manual  |
"""

import os
import time
import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Import our RAG components
from src.rag_pipeline import RAGPipeline, RAGResponse
from src.hybrid_search import HybridSearchEngine, SearchResult
from src.database_pool import DatabaseManager  # Now using pooled version!
from src.cache import CacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances (initialized on startup)
rag_pipeline = None
search_engine = None
db_manager = None
cache_manager = None


# Pydantic models for request/response validation
class QuestionRequest(BaseModel):
    """Request model for question answering."""
    question: str = Field(..., min_length=5, max_length=500)
    max_papers: int = Field(default=5, ge=1, le=20)
    min_relevance: float = Field(default=0.3, ge=0.0, le=1.0)
    
    @validator('question')
    def validate_question(cls, v):
        # Basic validation to ensure it's a question
        if not any(v.strip().endswith(c) for c in ['?', '.', '!']) and '?' not in v:
            v = v.strip() + '?'
        return v


class SearchRequest(BaseModel):
    """Request model for hybrid search."""
    query: str = Field(..., min_length=2, max_length=200)
    limit: int = Field(default=10, ge=1, le=50)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    database: str
    model: str
    papers_count: int
    embeddings_count: int


class QuestionResponse(BaseModel):
    """Response model for question answering."""
    question: str
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    response_time: float
    timestamp: datetime


class SearchResponse(BaseModel):
    """Response model for search."""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    response_time: float


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown.
    """
    # Startup
    global rag_pipeline, search_engine, db_manager, cache_manager
    
    logger.info("Starting Medical Literature RAG API...")
    
    try:
        # Initialize database manager with connection pooling
        logger.info("Initializing database with connection pooling...")
        db_manager = DatabaseManager(use_pool=True)  # Enable pooling for production!
        
        # Initialize cache manager
        logger.info("Initializing Redis cache...")
        cache_manager = CacheManager()
        cache_healthy, cache_msg = cache_manager.health_check()
        if not cache_healthy:
            logger.warning(f"Cache not available: {cache_msg}")
        
        # Initialize search engine
        logger.info("Loading search engine...")
        search_engine = HybridSearchEngine()
        
        # Initialize RAG pipeline
        logger.info("Loading RAG pipeline with Llama 3.1 8B Instruct model...")
        rag_pipeline = RAGPipeline()
        
        logger.info("✅ API startup complete!")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")
    if rag_pipeline:
        rag_pipeline.close()
    if search_engine:
        search_engine.close()
    if cache_manager:
        cache_manager.close()
    if db_manager:
        db_manager.close()
    logger.info("API shutdown complete")


# Create FastAPI app with lifespan manager
app = FastAPI(
    title="Medical Literature RAG API",
    description="AI-powered medical question answering using PubMed literature",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Medical Literature RAG API",
        "version": "1.0.0",
        "model": "Llama-3.1-8B-Instruct",
        "papers": "10 cancer research papers from PubMed",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify system status.
    """
    try:
        # Get database stats
        stats = db_manager.get_stats()
        
        # Check cache health
        cache_healthy = False
        if cache_manager:
            cache_healthy, _ = cache_manager.health_check()
        
        return HealthResponse(
            status="healthy" if cache_healthy else "healthy (cache unavailable)",
            timestamp=datetime.now(),
            database="connected",
            model="Llama-3.1-8B-Instruct loaded" if rag_pipeline and rag_pipeline.llm else "Model not loaded",
            papers_count=stats['total_papers'],
            embeddings_count=stats['papers_with_embeddings']
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint for medical question answering.
    
    Takes a medical question and returns an answer with citations.
    Uses Redis caching to achieve <2 second response times for repeated queries.
    """
    start_time = time.time()
    
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        # Get current paper count for cache invalidation
        paper_count = db_manager.get_stats()['total_papers'] if db_manager else None
        
        # Check cache first
        if cache_manager:
            cached_response = cache_manager.get(
                query=request.question,
                max_papers=request.max_papers,
                min_relevance=request.min_relevance,
                paper_count=paper_count
            )
            
            if cached_response:
                # Return cached response with updated timestamp
                cache_time = time.time() - start_time
                logger.info(f"✅ Returned cached response in {cache_time:.3f}s")
                
                return QuestionResponse(
                    question=request.question,
                    answer=cached_response['answer'],
                    citations=cached_response['citations'],
                    confidence=cached_response['confidence'],
                    response_time=cache_time,  # Report actual retrieval time
                    timestamp=datetime.now()
                )
        
        # Not in cache, generate new answer
        logger.info(f"Processing question: {request.question}")
        
        # Run the blocking LLM generation in a thread pool to avoid timeout
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(
                executor,
                rag_pipeline.generate_answer,
                request.question,
                request.max_papers,
                request.min_relevance
            )
        
        # Cache the response for future requests
        if cache_manager:
            cache_data = {
                'answer': response.answer,
                'citations': response.citations,
                'confidence': response.confidence,
                'generated_at': datetime.now().isoformat()
            }
            
            cache_success = cache_manager.set(
                query=request.question,
                response=cache_data,
                max_papers=request.max_papers,
                min_relevance=request.min_relevance,
                paper_count=paper_count
            )
            
            if cache_success:
                logger.info("✅ Response cached for future requests")
        
        return QuestionResponse(
            question=response.query,
            answer=response.answer,
            citations=response.citations,
            confidence=response.confidence,
            response_time=response.response_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_papers(request: SearchRequest):
    """
    Search for relevant papers using hybrid search.
    
    Combines vector similarity and keyword matching.
    """
    start_time = time.time()
    
    try:
        if not search_engine:
            raise HTTPException(status_code=503, detail="Search engine not initialized")
        
        # Perform hybrid search
        results = search_engine.hybrid_search(
            query=request.query,
            limit=request.limit,
            vector_weight=request.vector_weight,
            keyword_weight=request.keyword_weight
        )
        
        # Convert results to dictionaries
        results_dict = [r.to_dict() for r in results]
        
        return SearchResponse(
            query=request.query,
            results=results_dict,
            total_results=len(results),
            response_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Error searching papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/papers/{pmid}")
async def get_paper(pmid: str):
    """
    Get details of a specific paper by PMID.
    """
    try:
        paper = db_manager.get_paper_by_pmid(pmid)
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper with PMID {pmid} not found")
        
        # Convert date to string if needed
        if paper.get('pub_date'):
            paper['pub_date'] = str(paper['pub_date'])
        
        return paper
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching paper {pmid}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_statistics():
    """
    Get system statistics and metrics.
    """
    try:
        stats = db_manager.get_stats()
        
        # Get cache statistics
        cache_stats = cache_manager.get_stats() if cache_manager else {}
        
        return {
            "total_papers": stats['total_papers'],
            "papers_with_embeddings": stats['papers_with_embeddings'],
            "unique_journals": stats['unique_journals'],
            "date_range": {
                "earliest": str(stats['earliest_paper']) if stats['earliest_paper'] else None,
                "latest": str(stats['latest_paper']) if stats['latest_paper'] else None
            },
            "model": {
                "name": "Llama-3.1-8B-Instruct",
                "context_window": 8192,
                "embedding_dimensions": 768
            },
            "search_weights": {
                "vector": 0.7,
                "keyword": 0.3
            },
            "cache": {
                "hits": cache_stats.get('hits', 0),
                "misses": cache_stats.get('misses', 0),
                "hit_rate": f"{cache_stats.get('hit_rate', 0):.1%}",
                "cached_queries": cache_stats.get('cached_queries', 0),
                "time_saved_seconds": cache_stats.get('total_time_saved', 0),
                "memory_used_mb": round(cache_stats.get('redis_memory_used_mb', 0), 2)
            } if cache_stats else None
        }
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
    """
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    # Run server for development
    run_server(host="127.0.0.1", port=8000, reload=False)