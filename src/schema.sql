-- Medical Literature RAG Database Schema
-- This file defines the structure of our PostgreSQL database , "designs the rooms and the furniture"
-- for storing and querying medical research papers from PubMed

-- Enable pgvector extension for vector operations
CREATE EXTENSION IF NOT EXISTS vector;

-- Main papers table
CREATE TABLE IF NOT EXISTS papers (
    -- Primary key
    id SERIAL PRIMARY KEY,
    
    -- PubMed identifier (unique)
    pmid VARCHAR(20) UNIQUE NOT NULL,
    
    -- Paper metadata
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT[], -- Array of author names
    journal VARCHAR(500),
    pub_date DATE,
    mesh_terms TEXT[], -- Medical Subject Headings
    doi VARCHAR(200),
    
    -- Vector embedding for semantic search
    -- 384 dimensions for all-MiniLM-L6-v2 model
    -- We'll use 768 for better models later
    embedding vector(384),
    
    -- Full-text search vector for keyword search
    search_vector tsvector,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
-- Note: pmid already has an index from UNIQUE constraint, no need for another

-- Index for date-based queries
CREATE INDEX IF NOT EXISTS idx_papers_pub_date ON papers(pub_date);

-- Index for full-text search (GIN = Generalized Inverted Index)
CREATE INDEX IF NOT EXISTS idx_papers_search ON papers USING GIN(search_vector);

-- Index for vector similarity search (using IVFFlat method)
-- We'll create this after we have data, as it needs to be tuned
-- CREATE INDEX idx_papers_embedding ON papers USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Function to automatically update the search_vector when paper is inserted/updated
CREATE OR REPLACE FUNCTION update_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := 
        setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.abstract, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(array_to_string(NEW.mesh_terms, ' '), '')), 'C');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to call the function
CREATE TRIGGER update_papers_search_vector
BEFORE INSERT OR UPDATE ON papers
FOR EACH ROW
EXECUTE FUNCTION update_search_vector();

-- Table for storing query history (useful for analytics)
CREATE TABLE IF NOT EXISTS query_history (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding vector(384),
    result_count INTEGER,
    response_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Comments for documentation
COMMENT ON TABLE papers IS 'Stores medical research papers from PubMed';
COMMENT ON COLUMN papers.pmid IS 'PubMed ID - unique identifier from PubMed';
COMMENT ON COLUMN papers.embedding IS 'Vector embedding for semantic similarity search';
COMMENT ON COLUMN papers.search_vector IS 'Full-text search index for keyword matching';
COMMENT ON COLUMN papers.mesh_terms IS 'Medical Subject Headings - standardized medical keywords';