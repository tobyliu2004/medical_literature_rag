"""
Database connection and operations for Medical Literature RAG.
Handles all PostgreSQL interactions including vector operations.
AKA the butler who managges the database house.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.extensions import register_adapter
from pgvector.psycopg2 import register_vector
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "medical_rag"),
    "user": os.getenv("DB_USER", "rag_user"),
    "password": os.getenv("DB_PASSWORD", "rag_password")
}


class DatabaseManager:
    """Manages all database operations for the Medical Literature RAG system."""
    
    def __init__(self):
        """Initialize database connection."""
        self.config = DB_CONFIG
        # Register pgvector type with psycopg2
        self.connection = None
        self.connect()
    
    def connect(self):
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(**self.config)
            # Register vector type for pgvector operations
            register_vector(self.connection)
            print("✓ Connected to PostgreSQL database")
        except psycopg2.Error as e:
            print(f"✗ Failed to connect to database: {e}")
            raise
    
    @contextmanager
    def get_cursor(self, dict_cursor=True):
        """
        Context manager for database cursor.
        Automatically handles commit/rollback and cursor closing.
        """
        cursor_factory = RealDictCursor if dict_cursor else None
        cursor = self.connection.cursor(cursor_factory=cursor_factory)
        try:
            yield cursor
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            print(f"✗ Database error: {e}")
            raise
        finally:
            cursor.close()
    
    def insert_paper(self, paper: Dict[str, Any]) -> Optional[int]:
        """
        Insert a single paper into the database.
        
        Args:
            paper: Dictionary containing paper data
            
        Returns:
            ID of inserted paper, or None if failed
        """
        query = """
            INSERT INTO papers (
                pmid, title, abstract, authors, journal, 
                pub_date, mesh_terms, doi
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (pmid) DO UPDATE SET
                title = EXCLUDED.title,
                abstract = EXCLUDED.abstract,
                authors = EXCLUDED.authors,
                journal = EXCLUDED.journal,
                pub_date = EXCLUDED.pub_date,
                mesh_terms = EXCLUDED.mesh_terms,
                doi = EXCLUDED.doi,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id;
        """
        
        try:
            with self.get_cursor() as cursor:
                # Convert pub_date string to date object
                pub_date = None
                if paper.get('pub_date'):
                    try:
                        # Handle year-only dates
                        if len(paper['pub_date']) == 4:
                            pub_date = f"{paper['pub_date']}-01-01"
                        else:
                            pub_date = paper['pub_date']
                    except (ValueError, TypeError, AttributeError):
                        # ValueError: invalid date format
                        # TypeError: pub_date is not a string
                        # AttributeError: pub_date has no len()
                        pub_date = None
                
                cursor.execute(query, (
                    paper.get('pmid'),
                    paper.get('title'),
                    paper.get('abstract'),
                    paper.get('authors', []),
                    paper.get('journal'),
                    pub_date,
                    paper.get('mesh_terms', []),
                    paper.get('doi')
                ))
                
                result = cursor.fetchone()
                return result['id'] if result else None
                
        except psycopg2.Error as e:
            print(f"✗ Failed to insert paper {paper.get('pmid')}: {e}")
            return None
    
    def insert_papers_batch(self, papers: List[Dict[str, Any]]) -> int:
        """
        Insert multiple papers in a batch.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Number of papers successfully inserted
        """
        count = 0
        for paper in papers:
            if self.insert_paper(paper):
                count += 1
        
        print(f"✓ Inserted {count}/{len(papers)} papers")
        return count
    
    def get_paper_by_pmid(self, pmid: str) -> Optional[Dict]:
        """Fetch a paper by its PubMed ID."""
        query = "SELECT * FROM papers WHERE pmid = %s;"
        
        with self.get_cursor() as cursor:
            cursor.execute(query, (pmid,))
            return cursor.fetchone()
    
    def search_papers(self, search_term: str, limit: int = 10) -> List[Dict]:
        """
        Simple text search for papers (using PostgreSQL full-text search).
        This is a basic search - we'll add vector search later.
        
        Args:
            search_term: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of matching papers
        """
        query = """
            SELECT 
                pmid, title, abstract, journal, pub_date,
                ts_rank(search_vector, plainto_tsquery('english', %s)) as rank
            FROM papers
            WHERE search_vector @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s;
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(query, (search_term, search_term, limit))
            return cursor.fetchall()
    
    def update_embedding(self, pmid: str, embedding: np.ndarray):
        """
        Update the vector embedding for a paper.
        
        Args:
            pmid: PubMed ID
            embedding: Numpy array of embedding values
        """
        query = """
            UPDATE papers 
            SET embedding = %s, updated_at = CURRENT_TIMESTAMP
            WHERE pmid = %s;
        """
        
        with self.get_cursor(dict_cursor=False) as cursor:
            # Convert numpy array to list for pgvector
            embedding_list = embedding.tolist()
            cursor.execute(query, (embedding_list, pmid))
    
    def vector_search(self, query_embedding: np.ndarray, limit: int = 10) -> List[Dict]:
        """
        Search for papers using vector similarity.
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            
        Returns:
            List of similar papers with similarity scores
        """
        query = """
            SELECT 
                pmid, title, abstract, journal, pub_date,
                1 - (embedding <=> %s) as similarity
            FROM papers
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s
            LIMIT %s;
        """
        
        with self.get_cursor() as cursor:
            embedding_list = query_embedding.tolist()
            cursor.execute(query, (embedding_list, embedding_list, limit))
            return cursor.fetchall()
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        stats_query = """
            SELECT 
                COUNT(*) as total_papers,
                COUNT(embedding) as papers_with_embeddings,
                COUNT(DISTINCT journal) as unique_journals,
                MIN(pub_date) as earliest_paper,
                MAX(pub_date) as latest_paper
            FROM papers;
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(stats_query)
            return cursor.fetchone()
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            print("✓ Database connection closed")


def test_connection():
    """Test database connection and basic operations."""
    print("\n" + "="*60)
    print("Testing Database Connection")
    print("="*60)
    
    try:
        # Initialize database manager
        db = DatabaseManager()
        
        # Get statistics
        stats = db.get_stats()
        print(f"\nDatabase Statistics:")
        print(f"  Total papers: {stats['total_papers']}")
        print(f"  Papers with embeddings: {stats['papers_with_embeddings']}")
        print(f"  Unique journals: {stats['unique_journals']}")
        
        # Close connection
        db.close()
        
        print("\n✅ Database test successful!")
        
    except Exception as e:
        print(f"\n❌ Database test failed: {e}")


if __name__ == "__main__":
    test_connection()