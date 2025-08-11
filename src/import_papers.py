"""
Import papers from JSON to PostgreSQL database.
This script migrates our existing JSON data to the database.
"""

import json
import sys
from pathlib import Path
from database import DatabaseManager


def import_json_to_db(json_file: str):
    """
    Import papers from JSON file to database.
    
    Args:
        json_file: Path to JSON file containing papers
    """
    print(f"\n{'='*60}")
    print("Importing Papers to Database")
    print(f"{'='*60}")
    
    # Check if file exists
    if not Path(json_file).exists():
        print(f"❌ File not found: {json_file}")
        return
    
    # Load papers from JSON
    print(f"📂 Loading papers from: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    print(f"📊 Found {len(papers)} papers to import")
    
    # Initialize database
    try:
        db = DatabaseManager()
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        print("\nMake sure Docker is running:")
        print("  docker-compose up -d")
        return
    
    # Get initial stats
    stats_before = db.get_stats()
    print(f"\nDatabase before import:")
    print(f"  Papers: {stats_before['total_papers']}")
    
    # Import papers
    print(f"\n📥 Importing papers...")
    success_count = db.insert_papers_batch(papers)
    
    # Get final stats
    stats_after = db.get_stats()
    print(f"\nDatabase after import:")
    print(f"  Papers: {stats_after['total_papers']}")
    print(f"  New papers added: {stats_after['total_papers'] - stats_before['total_papers']}")
    
    # Test search functionality
    print(f"\n🔍 Testing search functionality...")
    
    # Test keyword search
    search_results = db.search_papers("immunotherapy", limit=3)
    if search_results:
        print(f"\nSearch for 'immunotherapy' found {len(search_results)} results:")
        for i, paper in enumerate(search_results, 1):
            print(f"  {i}. {paper['title'][:60]}...")
            print(f"     Relevance score: {paper['rank']:.3f}")
    else:
        print("  No results found (search index may need time to build)")
    
    # Close connection
    db.close()
    
    print(f"\n{'='*60}")
    print(f"✅ Import complete! {success_count} papers imported.")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Default to our sample data
    json_file = "data/cancer_papers_sample.json"
    
    # Allow custom file via command line
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    
    import_json_to_db(json_file)