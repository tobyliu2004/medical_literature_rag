"""
RAG (Retrieval-Augmented Generation) pipeline using Llama.
Takes top 5 retrieved papers and generates answers with citations for chats user asks
Creates a comprehensive answer based on medical literature with citations
Handles edge cases and rejects irrelevant queries such as "What is the recipe for chocolate cake?"
Minimu relevance score of 0.3 for papers

  3. Third: Add 25K AI/ML papers (bulk expansion)
  4. Fourth: Implement dynamic fetcher
  5. Later: Set up quarterly updates (after deployment)
"""

import os
import time
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Try to import llama-cpp-python, fallback to mock for testing
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("Warning: llama-cpp-python not available, using mock responses")

from src.hybrid_search import HybridSearchEngine, SearchResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"  # Llama 3.1 8B - Better instruction following
MODEL_PARAMS = {
    'n_ctx': 8192,  # Increased context window (Llama 3.1 supports 128K but we'll use 8K for speed)
    'n_threads': 8,  # Use 8 threads on M3 Pro
    'n_gpu_layers': 28,  # Optimized for M3 Pro (testing max layers)
    'temperature': 0.1,  # Very low for consistency and citation accuracy
    'top_p': 0.95,
    'max_tokens': 512,
    'repeat_penalty': 1.1  # Prevent repetitive citations
}


@dataclass
class RAGResponse:
    """Represents a generated response with citations."""
    query: str
    answer: str
    citations: List[Dict]
    confidence: float
    response_time: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses."""
        return {
            'query': self.query,
            'answer': self.answer,
            'citations': self.citations,
            'confidence': self.confidence,
            'response_time_seconds': self.response_time,
            'timestamp': datetime.now().isoformat()
        }


class RAGPipeline:
    """Main RAG pipeline combining search and generation."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            model_path: Path to the Llama model file
        """
        self.model_path = model_path or MODEL_PATH
        self.llm = None
        self.search_engine = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize components."""
        # Initialize search engine
        logger.info("Initializing search engine...")
        self.search_engine = HybridSearchEngine()
        
        # Initialize LLM (skip if not available for testing)
        if LLAMA_AVAILABLE and os.path.exists(self.model_path):
            logger.info(f"Loading Llama model from {self.model_path}...")
            start_time = time.time()
            
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=MODEL_PARAMS['n_ctx'],
                n_threads=MODEL_PARAMS['n_threads'],
                n_gpu_layers=MODEL_PARAMS['n_gpu_layers'],
                verbose=False
            )
            
            load_time = time.time() - start_time
            logger.info(f"✓ Model loaded in {load_time:.2f} seconds")
        else:
            logger.warning("Llama model not available - using mock responses for testing")
            self.llm = None
    
    def _format_context(self, papers: List[SearchResult]) -> str:
        """
        Format retrieved papers into context for the LLM.
        
        Args:
            papers: List of search results
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, paper in enumerate(papers, 1):
            # Truncate abstract to save context space
            abstract = paper.abstract[:500] + "..." if len(paper.abstract) > 500 else paper.abstract
            
            context_parts.append(
                f"Paper {i}:\n"
                f"Title: {paper.title}\n"
                f"PMID: {paper.pmid}\n"
                f"Abstract: {abstract}\n"
            )
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build the prompt for the LLM.
        
        Args:
            query: User's question
            context: Formatted paper context
            
        Returns:
            Complete prompt for the LLM
        """
        # Llama 3.1 prompt format with system message
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical research assistant that MUST cite sources. Every medical fact requires a [PMID: number] citation.

CITATION RULES:
1. Use ONLY this format: [PMID: 12345678]
2. Place citations immediately after each claim
3. NEVER write "Answer:" at the beginning
4. If you cannot find a PMID for a claim, do not make that claim

EXAMPLE OUTPUT:
Immune checkpoint inhibitors like PD-1 antibodies show 20-30% response rates in melanoma [PMID: 34567890]. Combining with CTLA-4 inhibitors improves outcomes to 40-50% [PMID: 23456789].<|eot_id|>

<|start_header_id|>user<|end_header_id|>

Using ONLY the papers below, answer this question: {query}

AVAILABLE PAPERS:
{context}

Remember: Every claim needs [PMID: number] citation.<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>

"""
        
        return prompt
    
    def _extract_citations(self, answer: str, papers: List[SearchResult]) -> List[Dict]:
        """
        Extract citations from the generated answer.
        
        Args:
            answer: Generated answer text
            papers: Papers that were provided as context
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        # Create PMID to paper mapping
        pmid_map = {paper.pmid: paper for paper in papers}
        
        # Find all PMID references in the answer - handle multiple formats
        import re
        # Match [PMID: 12345] or [PMID: 12345, 67890] or just PMID: 12345
        pmid_pattern = r'PMID:\s*(\d+)'
        found_pmids = re.findall(pmid_pattern, answer)
        
        # Build citation list
        for pmid in set(found_pmids):  # Use set to avoid duplicates
            if pmid in pmid_map:
                paper = pmid_map[pmid]
                citations.append({
                    'pmid': pmid,
                    'title': paper.title,
                    'journal': paper.journal,
                    'relevance_score': paper.combined_score
                })
        
        return citations
    
    def _generate_with_llm(self, prompt: str) -> str:
        """
        Generate response using Llama model.
        
        Args:
            prompt: Complete prompt for the model
            
        Returns:
            Generated text
        """
        if self.llm:
            # Generate with actual model
            response = self.llm(
                prompt,
                max_tokens=MODEL_PARAMS['max_tokens'],
                temperature=MODEL_PARAMS['temperature'],
                top_p=MODEL_PARAMS['top_p'],
                echo=False  # Don't include prompt in response
            )
            return response['choices'][0]['text'].strip()
        else:
            # Mock response for testing
            return self._generate_mock_response(prompt)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate a mock response for testing without model."""
        # Extract query from prompt
        query_match = prompt.split("Question: ")[-1].split("\n")[0]
        
        # Generate a realistic-looking response
        if "immunotherapy" in query_match.lower():
            return """Based on the provided research papers, immunotherapy has shown significant promise in lung cancer treatment [PMID: 36810079]. 

The papers indicate that immune checkpoint inhibitors, particularly PD-1 and PD-L1 inhibitors, have become important treatment options for non-small cell lung cancer (NSCLC) [PMID: 37185393]. These treatments work by blocking proteins that prevent the immune system from attacking cancer cells.

Current immunotherapy approaches include:
1. Monotherapy with checkpoint inhibitors for advanced disease
2. Combination therapy with chemotherapy
3. Neoadjuvant immunotherapy before surgery [PMID: 37270700]

The effectiveness varies based on PD-L1 expression levels and other biomarkers. Response rates range from 20-45% depending on patient selection and treatment regimen."""
        else:
            return f"Based on the research papers provided, I can address your question about {query_match}. [PMID: 37226190] The available literature suggests several key findings relevant to your query."
    
    def generate_answer(self, 
                        query: str, 
                        max_papers: int = 5,
                        min_relevance: float = 0.3) -> RAGResponse:
        """
        Generate an answer to a query using RAG.
        
        Args:
            query: User's question
            max_papers: Maximum number of papers to use as context
            min_relevance: Minimum relevance score for papers
            
        Returns:
            RAGResponse object with answer and citations
        """
        logger.info(f"Processing query: '{query}'")
        start_time = time.time()
        
        # Step 1: Retrieve relevant papers
        logger.info("Retrieving relevant papers...")
        papers = self.search_engine.hybrid_search(query, limit=max_papers)
        
        # Filter by minimum relevance
        papers = [p for p in papers if p.combined_score >= min_relevance]
        
        if not papers:
            return RAGResponse(
                query=query,
                answer="I couldn't find relevant papers to answer your question. Please try rephrasing or asking about a different topic.",
                citations=[],
                confidence=0.0,
                response_time=time.time() - start_time
            )
        
        logger.info(f"Using {len(papers)} papers as context")
        
        # Step 2: Format context
        context = self._format_context(papers)
        
        # Step 3: Build prompt
        prompt = self._build_prompt(query, context)
        
        # Step 4: Generate answer
        logger.info("Generating answer...")
        answer = self._generate_with_llm(prompt)
        
        # Step 5: Extract citations
        citations = self._extract_citations(answer, papers)
        
        # Calculate confidence based on paper relevance
        avg_relevance = sum(p.combined_score for p in papers) / len(papers)
        confidence = min(avg_relevance * 1.2, 1.0)  # Scale up slightly, cap at 1.0
        
        response_time = time.time() - start_time
        logger.info(f"✓ Answer generated in {response_time:.2f} seconds")
        
        return RAGResponse(
            query=query,
            answer=answer,
            citations=citations,
            confidence=confidence,
            response_time=response_time
        )
    
    def close(self) -> None:
        """Clean up resources."""
        if self.search_engine:
            self.search_engine.close()
        self.llm = None
        logger.info("✓ RAG pipeline closed")


def demonstrate_rag():
    """Demonstrate the RAG pipeline capabilities."""
    print("\n" + "="*60)
    print("RAG PIPELINE DEMONSTRATION")
    print("="*60)
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Test queries
    test_queries = [
        "What are the current immunotherapy options for lung cancer?",
        "How is artificial intelligence being used in cancer diagnosis?",
        "What is the role of PD-L1 expression in treatment decisions?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Question: {query}")
        print("-"*60)
        
        # Generate answer
        response = pipeline.generate_answer(query)
        
        print(f"\nAnswer:")
        print(response.answer)
        
        print(f"\nCitations ({len(response.citations)}):")
        for citation in response.citations:
            print(f"  - {citation['title'][:60]}...")
            print(f"    PMID: {citation['pmid']}, Relevance: {citation['relevance_score']:.3f}")
        
        print(f"\nMetrics:")
        print(f"  Confidence: {response.confidence:.2%}")
        print(f"  Response time: {response.response_time:.2f} seconds")
    
    # Demonstrate handling of edge cases
    print(f"\n{'='*60}")
    print("EDGE CASE: Irrelevant Query")
    print("-"*60)
    
    irrelevant_query = "What is the recipe for chocolate cake?"
    response = pipeline.generate_answer(irrelevant_query)
    print(f"Question: {irrelevant_query}")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence:.2%}")
    
    pipeline.close()
    
    print(f"\n{'='*60}")
    print("✅ RAG PIPELINE READY")
    print("Note: Using mock responses since Llama model not downloaded yet")
    print("To use real model, download from: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")
    print("="*60)


if __name__ == "__main__":
    demonstrate_rag()