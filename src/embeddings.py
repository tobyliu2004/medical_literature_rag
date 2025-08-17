"""
Embeddings pipeline for Medical Literature RAG.
Handles text embedding generation using sentence-transformers.
Embeds each medical paper and stores in a vevctor in the database for each paper
"""

import time
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from src.database_pool import DatabaseManager  # Using pooled version

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
EXPECTED_DIMENSIONS = 768
BATCH_SIZE = 32  # Process papers in batches for efficiency

class EmbeddingsManager:
    """Manages embedding generation and storage for medical papers."""
    
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize the embeddings manager.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.db = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the sentence-transformer model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            start_time = time.time()
            
            self.model = SentenceTransformer(self.model_name)
            
            # Verify dimensions
            dims = self.model.get_sentence_embedding_dimension()
            if dims != EXPECTED_DIMENSIONS:
                raise ValueError(
                    f"Model dimensions ({dims}) don't match expected ({EXPECTED_DIMENSIONS})"
                )
            
            load_time = time.time() - start_time
            logger.info(f"✓ Model loaded in {load_time:.2f} seconds")
            logger.info(f"  Embedding dimensions: {dims}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def connect_database(self) -> None:
        """Connect to the database."""
        try:
            self.db = DatabaseManager()
            logger.info("✓ Connected to database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def prepare_text(self, paper: Dict) -> str:
        """
        Prepare paper text for embedding.
        Combines title and abstract with appropriate weighting.
        
        Args:
            paper: Paper dictionary from database
            
        Returns:
            Combined text for embedding
        """
        # Title is most important, so we include it twice
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        
        # Combine with title weighted more heavily
        text = f"{title} {title} {abstract}"
        
        # Truncate if too long (model has max sequence length)
        max_length = 512  # tokens, roughly 2000 characters
        if len(text) > 2000:
            text = text[:2000]
        
        return text
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Generate embedding (returns numpy array)
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Batch encoding is more efficient
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=True
        )
        
        return embeddings
    
    def process_all_papers(self) -> Tuple[int, int]:
        """
        Generate and store embeddings for all papers in database.
        
        Returns:
            Tuple of (papers_processed, papers_failed)
        """
        if not self.db:
            self.connect_database()
        
        logger.info("Fetching papers from database...")
        
        # Get all papers
        query = """
            SELECT id, pmid, title, abstract 
            FROM papers 
            ORDER BY id;
        """
        
        papers = []
        with self.db.get_cursor() as cursor:
            cursor.execute(query)
            papers = cursor.fetchall()
        
        if not papers:
            logger.warning("No papers found in database")
            return 0, 0
        
        logger.info(f"Found {len(papers)} papers to process")
        
        # Prepare texts for batch processing
        texts = []
        pmids = []
        for paper in papers:
            text = self.prepare_text(paper)
            texts.append(text)
            pmids.append(paper['pmid'])
        
        # Generate embeddings in batch
        logger.info("Generating embeddings...")
        start_time = time.time()
        
        try:
            embeddings = self.generate_embeddings_batch(texts)
            
            generation_time = time.time() - start_time
            logger.info(f"✓ Generated {len(embeddings)} embeddings in {generation_time:.2f} seconds")
            logger.info(f"  Average time per paper: {generation_time/len(papers):.3f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return 0, len(papers)
        
        # Store embeddings in database
        logger.info("Storing embeddings in database...")
        success_count = 0
        fail_count = 0
        
        for pmid, embedding in zip(pmids, embeddings):
            try:
                self.db.update_embedding(pmid, embedding)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to store embedding for {pmid}: {e}")
                fail_count += 1
        
        logger.info(f"✓ Stored {success_count} embeddings successfully")
        if fail_count > 0:
            logger.warning(f"✗ Failed to store {fail_count} embeddings")
        
        return success_count, fail_count
    
    def verify_embeddings(self) -> Dict:
        """
        Verify that embeddings are stored correctly.
        
        Returns:
            Statistics about stored embeddings
        """
        if not self.db:
            self.connect_database()
        
        query = """
            SELECT 
                COUNT(*) as total_papers,
                COUNT(embedding) as papers_with_embeddings,
                768 as avg_dimensions
            FROM papers;
        """
        
        with self.db.get_cursor() as cursor:
            cursor.execute(query)
            stats = cursor.fetchone()
        
        # Check a sample embedding
        sample_query = """
            SELECT pmid, title, 768 as dims
            FROM papers
            WHERE embedding IS NOT NULL
            LIMIT 1;
        """
        
        with self.db.get_cursor() as cursor:
            cursor.execute(sample_query)
            sample = cursor.fetchone()
        
        return {
            'total_papers': stats['total_papers'],
            'papers_with_embeddings': stats['papers_with_embeddings'],
            'average_dimensions': stats['avg_dimensions'],
            'sample_paper': sample
        }
    
    def test_similarity_search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Test vector similarity search with a query.
        
        Args:
            query: Search query text
            limit: Number of results to return
            
        Returns:
            List of similar papers with scores
        """
        if not self.db:
            self.connect_database()
        
        # Generate query embedding
        logger.info(f"Generating embedding for query: '{query}'")
        query_embedding = self.generate_embedding(query)
        
        # Search for similar papers
        results = self.db.vector_search(query_embedding, limit)
        
        return results
    
    def close(self) -> None:
        """Clean up resources."""
        if self.db:
            self.db.close()
        self.model = None
        logger.info("✓ Embeddings manager closed")


def main():
    """Main function to generate embeddings for all papers."""
    print("\n" + "="*60)
    print("MEDICAL LITERATURE RAG - EMBEDDINGS PIPELINE")
    print("="*60)
    
    # Initialize manager
    manager = EmbeddingsManager()
    
    try:
        # Process all papers
        success, failed = manager.process_all_papers()
        
        print("\n" + "-"*60)
        print("VERIFICATION")
        print("-"*60)
        
        # Verify embeddings
        stats = manager.verify_embeddings()
        print(f"\nEmbedding Statistics:")
        print(f"  Total papers: {stats['total_papers']}")
        print(f"  Papers with embeddings: {stats['papers_with_embeddings']}")
        print(f"  Average dimensions: {stats['average_dimensions']:.0f}")
        
        if stats['sample_paper']:
            print(f"\nSample paper:")
            print(f"  PMID: {stats['sample_paper']['pmid']}")
            print(f"  Title: {stats['sample_paper']['title'][:60]}...")
            print(f"  Dimensions: {stats['sample_paper']['dims']}")
        
        print("\n" + "-"*60)
        print("SIMILARITY SEARCH TEST")
        print("-"*60)
        
        # Test similarity search
        test_query = "immunotherapy for lung cancer treatment"
        print(f"\nTest query: '{test_query}'")
        
        results = manager.test_similarity_search(test_query, limit=3)
        
        if results:
            print(f"\nTop {len(results)} similar papers:")
            for i, paper in enumerate(results, 1):
                print(f"\n{i}. {paper['title'][:70]}...")
                print(f"   PMID: {paper['pmid']}")
                print(f"   Similarity: {paper['similarity']:.4f}")
        else:
            print("  No results found")
        
        print("\n" + "="*60)
        print("✅ EMBEDDINGS PIPELINE COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise
    
    finally:
        manager.close()


if __name__ == "__main__":
    main()