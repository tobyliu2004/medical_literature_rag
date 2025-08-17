#!/usr/bin/env python3
"""
FINAL SCRIPT: Download 50,000 papers from PubMed for Medical Literature RAG.
This is the production script - run this once to populate your database.

Expected runtime: 2-3 hours
Can be interrupted and resumed (tracks what's already downloaded)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pubmed_bulk_download import PubMedDownloader
from database_pool import DatabaseManager
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def main():
    """Download 50,000 papers for production system."""
    
    TARGET = 50000  # Our goal
    
    print("\n" + "="*70)
    print(" MEDICAL LITERATURE RAG - 50,000 PAPER DOWNLOAD")
    print("="*70)
    print("\nThis will download 50,000 recent medical papers from PubMed")
    print("Topics: Lung cancer, immunotherapy, CAR-T, biomarkers, AI in oncology")
    print("Expected time: 2-3 hours")
    print("Can be safely interrupted and resumed\n")
    
    # Initialize database
    db = DatabaseManager(use_pool=True)
    
    # Check current status
    with db.get_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) as count FROM papers")
        current_count = cursor.fetchone()['count']
    
    print(f"Current papers in database: {current_count:,}")
    
    if current_count >= TARGET:
        print(f"✅ Already have {TARGET:,} papers! Download complete.")
        return
    
    papers_needed = TARGET - current_count
    print(f"Papers still needed: {papers_needed:,}")
    
    # Confirm before starting
    response = input(f"\nReady to download {papers_needed:,} papers? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled")
        return
    
    print("\nStarting download...")
    print("-"*70)
    
    # Start download
    start_time = time.time()
    downloader = PubMedDownloader(db)
    
    try:
        # This handles everything - batching, retries, progress
        downloader.download_papers(target_count=TARGET)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        print("Run this script again to resume where you left off")
        
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        print("Run this script again to resume")
        
    finally:
        # Final statistics
        with db.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM papers")
            final_count = cursor.fetchone()['count']
        
        elapsed = time.time() - start_time
        added = final_count - current_count
        
        print("\n" + "="*70)
        print(" DOWNLOAD SUMMARY")
        print("="*70)
        print(f"Papers at start:  {current_count:,}")
        print(f"Papers now:       {final_count:,}")
        print(f"Papers added:     {added:,}")
        print(f"Time elapsed:     {elapsed/60:.1f} minutes")
        
        if added > 0:
            print(f"Average speed:    {added/(elapsed/60):.0f} papers/minute")
        
        if final_count >= TARGET:
            print(f"\n🎉 SUCCESS! You now have {final_count:,} papers!")
            print("\nNext steps:")
            print("1. Run: python src/embeddings.py  (generates vectors, ~1 hour)")
            print("2. Your system is ready for production use!")
        else:
            remaining = TARGET - final_count
            print(f"\n📊 Progress: {final_count:,} / {TARGET:,} papers")
            print(f"Still need {remaining:,} more papers")
            print("Run this script again to continue downloading")
        
        print("="*70)
        
        db.close()


if __name__ == "__main__":
    main()