"""
Hybrid search combining vector similarity and keyword matching.
Gets the best of both worlds: semantic understanding + exact term matching.
70% vector search, 30% keyword search, combines the two approaches to find the most relevant papers for each request
Actually takes the chat request and retrieves the most relevant papers from the database
"""

import time
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from src.database_pool import DatabaseManager  # Using pooled version

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Search configuration
VECTOR_WEIGHT = 0.7  # 70% weight to semantic similarity
KEYWORD_WEIGHT = 0.3  # 30% weight to keyword matching
DEFAULT_LIMIT = 10


@dataclass
class SearchResult:
    """Represents a search result with combined scoring."""
    pmid: str
    title: str
    abstract: str
    journal: Optional[str]
    pub_date: Optional[str]
    vector_score: float
    keyword_score: float
    combined_score: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses."""
        return {
            'pmid': self.pmid,
            'title': self.title,
            'abstract': self.abstract[:500] + '...' if len(self.abstract) > 500 else self.abstract,
            'journal': self.journal,
            'pub_date': str(self.pub_date) if self.pub_date else None,
            'scores': {
                'vector': round(self.vector_score, 4),
                'keyword': round(self.keyword_score, 4),
                'combined': round(self.combined_score, 4)
            }
        }


class HybridSearchEngine:
    """Combines vector and keyword search for optimal retrieval."""
    
    def __init__(self):
        """Initialize the hybrid search engine."""
        self.model = None
        self.db = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize model and database connection."""
        try:
            # Load embedding model
            logger.info("Loading embedding model...")
            start_time = time.time()
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            load_time = time.time() - start_time
            logger.info(f"✓ Model loaded in {load_time:.2f} seconds")
            
            # Connect to database
            self.db = DatabaseManager()
            logger.info("✓ Connected to database")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
    
    def vector_search(self, query: str, limit: int = 20) -> Dict[str, float]:
        """
        Perform vector similarity search.
        
        Args:
            query: Search query
            limit: Maximum results (we get more than needed for re-ranking)
            
        Returns:
            Dictionary mapping PMID to similarity score
        """
        # Generate query embedding
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Search using vector similarity
        results = self.db.vector_search(query_embedding, limit)
        
        # Create score mapping
        scores = {}
        for paper in results:
            scores[paper['pmid']] = paper.get('similarity', paper.get('score', 0))
        
        return scores
    
    def keyword_search(self, query: str, limit: int = 20) -> Dict[str, float]:
        """
        Perform keyword/full-text search.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            Dictionary mapping PMID to relevance score
        """
        results = self.db.search_papers(query, limit)
        
        # Normalize scores (PostgreSQL rank can vary widely)
        scores = {}
        max_rank = max([r.get('rank', r.get('score', 0)) for r in results]) if results else 1.0
        
        for paper in results:
            # Normalize to 0-1 range
            raw_score = paper.get('rank', paper.get('score', 0))
            normalized_score = raw_score / max_rank if max_rank > 0 else 0
            scores[paper['pmid']] = normalized_score
        
        return scores
    
    def _preprocess_query_for_keyword_search(self, query: str) -> str:
        """
        Preprocess query to extract key medical terms for better keyword matching.
        Removes question words and focuses on medical/scientific terms.
        """
        # Remove common question words that dilute keyword search
        stop_phrases = [
            'what are', 'what is', 'how does', 'how do', 'when should',
            'why is', 'why are', 'can you', 'could you', 'explain',
            'describe', 'list', 'the', 'and their', 'specific'
        ]
        
        processed = query.lower()
        for phrase in stop_phrases:
            processed = processed.replace(phrase, '')
        
        # Remove punctuation
        processed = processed.replace('?', '').replace(',', '').replace('.', '')
        
        # Remove extra spaces
        processed = ' '.join(processed.split())
        
        return processed.strip()
    
    def hybrid_search(self, 
                     query: str, 
                     limit: int = DEFAULT_LIMIT,
                     vector_weight: float = VECTOR_WEIGHT,
                     keyword_weight: float = KEYWORD_WEIGHT) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and keyword methods.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            vector_weight: Weight for vector similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
            
        Returns:
            List of SearchResult objects sorted by combined score
        """
        # Parameter validation
        if limit < 0:
            logger.warning(f"Invalid limit {limit}, using default {DEFAULT_LIMIT}")
            limit = DEFAULT_LIMIT
        elif limit == 0:
            # Allow limit=0 to return empty results
            return []
        elif limit > 1000:
            logger.warning(f"Limit {limit} too high, capping at 1000")
            limit = 1000
            
        logger.info(f"Hybrid search for: '{query}'")
        start_time = time.time()
        
        # Ensure weights sum to 1
        total_weight = vector_weight + keyword_weight
        vector_weight = vector_weight / total_weight
        keyword_weight = keyword_weight / total_weight
        
        # Get results from both search methods
        # Fetch many more papers to ensure overlap between vector and keyword results
        # This is critical for research accuracy - we need papers that score well in BOTH methods
        fetch_multiplier = 10  # Fetch 10x more papers than requested
        max_fetch = min(limit * fetch_multiplier, 100)  # Cap at 100 for performance
        
        # Use full query for vector search (understands context)
        vector_scores = self.vector_search(query, max_fetch)
        
        # Use preprocessed query for keyword search (focuses on key terms)
        keyword_query = self._preprocess_query_for_keyword_search(query)
        keyword_scores = self.keyword_search(keyword_query, max_fetch)
        
        # Get all unique PMIDs
        all_pmids = set(vector_scores.keys()) | set(keyword_scores.keys())
        
        # Calculate combined scores
        combined_results = []
        
        for pmid in all_pmids:
            # Get scores (default to 0 if not found in one method)
            v_score = vector_scores.get(pmid, 0.0)
            k_score = keyword_scores.get(pmid, 0.0)
            
            # Calculate weighted combination
            combined = (v_score * vector_weight) + (k_score * keyword_weight)
            
            # Fetch paper details
            paper = self.db.get_paper_by_pmid(pmid)
            if paper:
                result = SearchResult(
                    pmid=pmid,
                    title=paper['title'],
                    abstract=paper['abstract'],
                    journal=paper.get('journal'),
                    pub_date=paper.get('pub_date'),
                    vector_score=v_score,
                    keyword_score=k_score,
                    combined_score=combined
                )
                combined_results.append(result)
        
        # Sort by combined score (highest first)
        combined_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Limit results
        combined_results = combined_results[:limit]
        
        search_time = time.time() - start_time
        logger.info(f"✓ Hybrid search completed in {search_time:.2f} seconds")
        logger.info(f"  Found {len(combined_results)} results")
        
        return combined_results
    
    def explain_search(self, query: str, pmid: str) -> Dict:
        """
        Explain why a particular paper was returned for a query.
        
        Args:
            query: The search query
            pmid: The paper's PubMed ID
            
        Returns:
            Explanation of scoring
        """
        # Get both scores
        vector_scores = self.vector_search(query, 50)
        keyword_scores = self.keyword_search(query, 50)
        
        v_score = vector_scores.get(pmid, 0.0)
        k_score = keyword_scores.get(pmid, 0.0)
        combined = (v_score * VECTOR_WEIGHT) + (k_score * KEYWORD_WEIGHT)
        
        # Get paper details
        paper = self.db.get_paper_by_pmid(pmid)
        
        explanation = {
            'query': query,
            'paper': {
                'pmid': pmid,
                'title': paper['title'] if paper else 'Unknown'
            },
            'scoring': {
                'vector_similarity': {
                    'score': round(v_score, 4),
                    'weight': VECTOR_WEIGHT,
                    'weighted_score': round(v_score * VECTOR_WEIGHT, 4),
                    'explanation': f"Semantic similarity between query and paper content"
                },
                'keyword_relevance': {
                    'score': round(k_score, 4),
                    'weight': KEYWORD_WEIGHT,
                    'weighted_score': round(k_score * KEYWORD_WEIGHT, 4),
                    'explanation': f"Full-text search relevance for query terms"
                },
                'combined_score': round(combined, 4)
            }
        }
        
        return explanation
    
    def close(self) -> None:
        """Clean up resources."""
        if self.db:
            self.db.close()
        logger.info("✓ Hybrid search engine closed")


def demonstrate_hybrid_search():
    """Demonstrate the hybrid search capabilities."""
    print("\n" + "="*60)
    print("HYBRID SEARCH DEMONSTRATION")
    print("="*60)
    
    engine = HybridSearchEngine()
    
    # Test queries showing different strengths
    test_queries = [
        {
            'query': 'PD-L1 expression in lung cancer',
            'description': 'Tests exact term (PD-L1) + semantic understanding'
        },
        {
            'query': 'immune checkpoint inhibitors for NSCLC treatment',
            'description': 'Tests semantic similarity (should find immunotherapy papers)'
        },
        {
            'query': 'artificial intelligence machine learning diagnosis',
            'description': 'Tests keyword matching for technical terms'
        }
    ]
    
    for test in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{test['query']}'")
        print(f"Purpose: {test['description']}")
        print("-"*60)
        
        # Perform hybrid search
        results = engine.hybrid_search(test['query'], limit=3)
        
        if results:
            print(f"\nTop {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result.title[:70]}...")
                print(f"   PMID: {result.pmid}")
                print(f"   Scores: Vector={result.vector_score:.3f}, "
                      f"Keyword={result.keyword_score:.3f}, "
                      f"Combined={result.combined_score:.3f}")
        else:
            print("No results found")
    
    # Show detailed explanation for one result
    print(f"\n{'='*60}")
    print("SCORING EXPLANATION")
    print("="*60)
    
    if results:
        explanation = engine.explain_search(test_queries[0]['query'], results[0].pmid)
        print(f"\nWhy did '{explanation['paper']['title'][:50]}...'")
        print(f"rank #1 for '{explanation['query']}'?")
        print("\nScoring breakdown:")
        
        for component, details in explanation['scoring'].items():
            if isinstance(details, dict) and 'score' in details:
                print(f"\n{component}:")
                print(f"  Raw score: {details['score']}")
                print(f"  Weight: {details['weight']}")
                print(f"  Contribution: {details['weighted_score']}")
    
    engine.close()
    
    print(f"\n{'='*60}")
    print("✅ HYBRID SEARCH READY FOR PRODUCTION")
    print("="*60)


if __name__ == "__main__":
    demonstrate_hybrid_search()