# Medical Literature RAG System

A retrieval-augmented generation (RAG) system for medical question-answering using cancer research papers from PubMed.

## Overview

This system allows medical professionals to ask natural language questions and receive accurate, cited answers from a database of 50,000+ cancer research papers.

## Project Status

🚧 **In Development** - Currently building the foundation

## Setup

### Prerequisites
- Python 3.11.8 (using conda ml-env)
- PostgreSQL (will be added later)
- 10GB free disk space for data and models

### Installation

1. Clone the repository
2. Copy environment template:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your email (required by PubMed)
4. Activate conda environment:
   ```bash
   conda activate ml-env
   ```

### Quick Start

Fetch your first cancer papers:
```bash
python fetch_pubmed.py
```

## Architecture

- **Data Source**: PubMed E-utilities API
- **Database**: PostgreSQL with pgvector extension
- **Search**: Hybrid (vector + keyword)
- **LLM**: Llama-3.1-8B (quantized)
- **API**: FastAPI
- **Caching**: Redis

## Development Log

- Step 1: Basic PubMed data fetching ✅
- Step 2: Database setup (coming next)
- Step 3: Embedding generation
- Step 4: Search implementation
- Step 5: RAG pipeline
- Step 6: API development

## License

MIT