-- Create IVFFlat index for 50K papers
-- Run this AFTER embeddings are complete, makes querying faster

-- First, check embedding count, verifies all papers have embeddings before creating index
SELECT COUNT(*) as papers_with_embeddings FROM papers WHERE embedding IS NOT NULL;

-- Create the vector index (this will take 2-3 minutes)
-- creates index concurrently to avoid locking the table named idx_papers_embedding_ivfflat
-- 200 groups (lists) for the IVFFlat index to look through
CREATE INDEX CONCURRENTLY idx_papers_embedding_ivfflat 
ON papers 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 200);

-- Create text search index for faster keyword search
CREATE INDEX CONCURRENTLY idx_papers_text_search 
ON papers 
USING GIN (to_tsvector('english', title || ' ' || abstract));

-- Analyze table for query optimizer, updates postgresql statistics about the table
ANALYZE papers;

-- Verify indexes were created
\di+ idx_papers*