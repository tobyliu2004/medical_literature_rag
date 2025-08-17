#!/usr/bin/env python3
"""
PROVIDES ALL COMPLEX LOGIC USED IN download_50k_papers.py to download 50,000 papers from PubMed.

PubMed Bulk Download Script for Medical Literature RAG.
Downloads up to 50,000 papers related to cancer, immunotherapy, and lung disease.

Strategy:
1. Use multiple search queries to get diverse, relevant papers
2. Download in batches to avoid API limits
3. Store incrementally to handle interruptions
4. Focus on recent papers (2020-2024) for current information

PubMed API limits:
- 3 requests per second without API key
- 10 requests per second with API key
- Max 10,000 results per search
"""

import time
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional
import os
from pathlib import Path

from Bio import Entrez
from src.database_pool import DatabaseManager
import psycopg2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PubMed API Configuration
Entrez.email = "your_email@example.com"  # Required by NCBI
BATCH_SIZE = 100  # Papers per API request
DELAY_BETWEEN_REQUESTS = 0.4  # Seconds (stay under rate limit)
MAX_PAPERS_PER_SEARCH = 10000  # PubMed limit

# Search queries for comprehensive coverage
SEARCH_QUERIES = [
    # Lung cancer and immunotherapy (primary focus)
    '("lung cancer"[Title/Abstract] AND "immunotherapy"[Title/Abstract]) AND ("2020"[Date - Publication] : "2024"[Date - Publication])',
    '("NSCLC"[Title/Abstract] AND ("PD-1" OR "PD-L1" OR "checkpoint")[Title/Abstract]) AND ("2020"[Date - Publication] : "2024"[Date - Publication])',
    '("lung adenocarcinoma"[Title/Abstract] OR "squamous cell carcinoma"[Title/Abstract]) AND ("2020"[Date - Publication] : "2024"[Date - Publication])',
    
    # CAR-T and cell therapy
    '("CAR-T"[Title/Abstract] OR "chimeric antigen receptor"[Title/Abstract]) AND "cancer"[Title/Abstract] AND ("2020"[Date - Publication] : "2024"[Date - Publication])',
    
    # Targeted therapy
    '("targeted therapy"[Title/Abstract] AND ("EGFR" OR "ALK" OR "ROS1")[Title/Abstract]) AND ("2020"[Date - Publication] : "2024"[Date - Publication])',
    
    # Biomarkers and diagnosis
    '("biomarker"[Title/Abstract] AND "lung cancer"[Title/Abstract]) AND ("2020"[Date - Publication] : "2024"[Date - Publication])',
    '("liquid biopsy"[Title/Abstract] OR "ctDNA"[Title/Abstract]) AND "cancer"[Title/Abstract] AND ("2020"[Date - Publication] : "2024"[Date - Publication])',
    
    # AI in oncology
    '("artificial intelligence"[Title/Abstract] OR "machine learning"[Title/Abstract]) AND "cancer diagnosis"[Title/Abstract] AND ("2020"[Date - Publication] : "2024"[Date - Publication])',
    
    # Clinical trials
    '("clinical trial"[Title/Abstract] AND "lung cancer"[Title/Abstract] AND "phase 3"[Title/Abstract]) AND ("2020"[Date - Publication] : "2024"[Date - Publication])',
    
    # Combination therapy
    '("combination therapy"[Title/Abstract] AND "immunotherapy"[Title/Abstract]) AND ("2020"[Date - Publication] : "2024"[Date - Publication])',
    
    # General cancer immunology
    '("tumor microenvironment"[Title/Abstract] OR "T cell"[Title/Abstract]) AND "cancer"[Title/Abstract] AND ("2021"[Date - Publication] : "2024"[Date - Publication])',
    '("checkpoint inhibitor"[Title/Abstract] AND ("nivolumab" OR "pembrolizumab" OR "atezolizumab")[Title/Abstract]) AND ("2020"[Date - Publication] : "2024"[Date - Publication])',
    
    # Resistance and adverse events
    '("resistance"[Title/Abstract] AND "immunotherapy"[Title/Abstract]) AND ("2020"[Date - Publication] : "2024"[Date - Publication])',
    '("adverse events"[Title/Abstract] OR "toxicity"[Title/Abstract]) AND "checkpoint inhibitor"[Title/Abstract] AND ("2020"[Date - Publication] : "2024"[Date - Publication])'
]


class PubMedDownloader:
    """Manages bulk download of PubMed papers."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize downloader.
        
        Args:
            db_manager: Database manager instance (creates new if None)
        """
        self.db = db_manager or DatabaseManager(use_pool=True)
        self.downloaded_pmids = self._get_existing_pmids()
        self.stats = {
            "searches_completed": 0,
            "papers_downloaded": 0,
            "papers_skipped": 0,
            "errors": 0,
            "start_time": time.time()
        }
    
    def _get_existing_pmids(self) -> Set[str]:
        """Get PMIDs already in database to avoid duplicates."""
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute("SELECT pmid FROM papers")
                results = cursor.fetchall()
                pmids = {row['pmid'] for row in results}
                logger.info(f"Found {len(pmids)} existing papers in database")
                return pmids
        except Exception as e:
            logger.error(f"Error getting existing PMIDs: {e}")
            return set()
    
    def search_pubmed(self, query: str, max_results: int = MAX_PAPERS_PER_SEARCH) -> List[str]:
        """
        Search PubMed and return PMIDs.
        
        Args:
            query: PubMed search query
            max_results: Maximum results to return
            
        Returns:
            List of PMIDs
        """
        try:
            logger.info(f"Searching: {query[:100]}...")
            
            # Perform search
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance",
                retmode="json"
            )
            
            results = json.loads(handle.read())
            handle.close()
            
            pmids = results.get('esearchresult', {}).get('idlist', [])
            logger.info(f"Found {len(pmids)} papers")
            
            time.sleep(DELAY_BETWEEN_REQUESTS)
            return pmids
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            self.stats["errors"] += 1
            return []
    
    def fetch_paper_details(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch detailed information for PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of paper dictionaries
        """
        papers = []
        
        try:
            # Fetch details
            handle = Entrez.efetch(
                db="pubmed",
                id=','.join(pmids),
                rettype="medline",
                retmode="text"
            )
            
            records = handle.read()
            handle.close()
            
            # Parse MEDLINE format
            current_paper = {}
            current_field = None
            
            for line in records.split('\n'):
                if line.startswith('PMID- '):
                    if current_paper:
                        papers.append(current_paper)
                    current_paper = {'pmid': line[6:].strip()}
                
                elif line.startswith('TI  - '):
                    current_paper['title'] = line[6:].strip()
                    current_field = 'title'
                
                elif line.startswith('AB  - '):
                    current_paper['abstract'] = line[6:].strip()
                    current_field = 'abstract'
                
                elif line.startswith('AU  - '):
                    if 'authors' not in current_paper:
                        current_paper['authors'] = []
                    current_paper['authors'].append(line[6:].strip())
                    current_field = 'authors'
                
                elif line.startswith('JT  - '):
                    current_paper['journal'] = line[6:].strip()
                    current_field = 'journal'
                
                elif line.startswith('DP  - '):
                    # Parse publication date
                    date_str = line[6:].strip()
                    try:
                        # Try to parse year at minimum
                        year = date_str.split()[0]
                        current_paper['pub_date'] = f"{year}-01-01"
                    except:
                        current_paper['pub_date'] = "2024-01-01"  # Default
                    current_field = None
                
                elif line.startswith('      ') and current_field:
                    # Continuation of previous field
                    if current_field == 'title':
                        current_paper['title'] += ' ' + line.strip()
                    elif current_field == 'abstract':
                        current_paper['abstract'] += ' ' + line.strip()
                
                elif not line.strip():
                    current_field = None
            
            # Don't forget the last paper
            if current_paper:
                papers.append(current_paper)
            
            time.sleep(DELAY_BETWEEN_REQUESTS)
            return papers
            
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            self.stats["errors"] += 1
            return []
    
    def save_papers_to_db(self, papers: List[Dict]) -> int:
        """
        Save papers to database.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Number of papers saved
        """
        saved = 0
        
        for paper in papers:
            # Skip if already in database
            if paper.get('pmid') in self.downloaded_pmids:
                self.stats["papers_skipped"] += 1
                continue
            
            # Ensure required fields
            if not paper.get('title') or not paper.get('abstract'):
                continue
            
            # Format authors as PostgreSQL array literal
            if 'authors' in paper:
                # PostgreSQL expects format: {author1,author2,author3}
                authors_list = paper['authors'][:10]  # Limit to 10 authors
                # Escape any commas or special chars in author names
                authors_list = [a.replace(',', ';').replace('{', '').replace('}', '') for a in authors_list]
                paper['authors'] = '{' + ','.join(authors_list) + '}'
            else:
                paper['authors'] = '{Unknown}'
            
            # Set defaults
            paper['journal'] = paper.get('journal', 'Unknown')
            paper['pub_date'] = paper.get('pub_date', '2024-01-01')
            
            # Save to database
            try:
                success = self.db.insert_paper(paper)
                if success:
                    saved += 1
                    self.downloaded_pmids.add(paper['pmid'])
                    self.stats["papers_downloaded"] += 1
                    
            except Exception as e:
                logger.error(f"Error saving paper {paper.get('pmid')}: {e}")
                self.stats["errors"] += 1
        
        return saved
    
    def download_papers(self, target_count: int = 50000) -> None:
        """
        Main download function.
        
        Args:
            target_count: Target number of papers to download
        """
        logger.info(f"Starting download. Target: {target_count} papers")
        logger.info(f"Already have {len(self.downloaded_pmids)} papers in database")
        
        all_pmids = set()
        
        # Execute searches
        for i, query in enumerate(SEARCH_QUERIES, 1):
            if len(self.downloaded_pmids) >= target_count:
                logger.info(f"Reached target of {target_count} papers")
                break
            
            logger.info(f"\n[{i}/{len(SEARCH_QUERIES)}] Executing search {i}...")
            
            # Search PubMed
            pmids = self.search_pubmed(query)
            
            # Filter out already downloaded
            new_pmids = [pmid for pmid in pmids if pmid not in self.downloaded_pmids]
            
            if not new_pmids:
                logger.info("No new papers from this search")
                continue
            
            # Process in batches
            for batch_start in range(0, len(new_pmids), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(new_pmids))
                batch_pmids = new_pmids[batch_start:batch_end]
                
                logger.info(f"Fetching batch {batch_start//BATCH_SIZE + 1} ({len(batch_pmids)} papers)...")
                
                # Fetch paper details
                papers = self.fetch_paper_details(batch_pmids)
                
                # Save to database
                saved = self.save_papers_to_db(papers)
                
                logger.info(f"Saved {saved} new papers. Total: {len(self.downloaded_pmids)}")
                
                # Check if target reached
                if len(self.downloaded_pmids) >= target_count:
                    break
            
            self.stats["searches_completed"] = i
        
        # Final statistics
        self._print_statistics()
    
    def _print_statistics(self):
        """Print download statistics."""
        elapsed = time.time() - self.stats["start_time"]
        
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE")
        print("="*60)
        print(f"Total papers in database:  {len(self.downloaded_pmids)}")
        print(f"Papers downloaded:         {self.stats['papers_downloaded']}")
        print(f"Papers skipped (duplicate): {self.stats['papers_skipped']}")
        print(f"Searches completed:        {self.stats['searches_completed']}")
        print(f"Errors:                    {self.stats['errors']}")
        print(f"Time elapsed:              {elapsed/60:.1f} minutes")
        
        if self.stats['papers_downloaded'] > 0:
            print(f"Average time per paper:    {elapsed/self.stats['papers_downloaded']:.2f} seconds")
        
        print("="*60)


def main():
    """Main function to download papers."""
    print("\n" + "="*60)
    print("PUBMED BULK DOWNLOAD FOR MEDICAL LITERATURE RAG")
    print("="*60)
    
    # Check current database status
    db = DatabaseManager(use_pool=True)
    
    with db.get_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) as count FROM papers")
        current_count = cursor.fetchone()['count']
    
    print(f"\nCurrent papers in database: {current_count}")
    
    # For initial test, let's download 1000 papers first
    TEST_TARGET = 1000  # Start with 1000 for testing
    
    if current_count >= TEST_TARGET:
        print(f"Already have {TEST_TARGET}+ papers. Proceeding to full 50K...")
        target = 50000
    else:
        print(f"Starting with test batch of {TEST_TARGET} papers...")
        target = TEST_TARGET
    
    # Calculate how many more papers needed
    papers_needed = target - current_count
    print(f"Papers needed: {papers_needed}")
    
    # Auto-confirm for demonstration
    print("\nStarting download...")
    
    # Start download
    downloader = PubMedDownloader(db)
    
    try:
        downloader.download_papers(target_count=50000)
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        downloader._print_statistics()
    except Exception as e:
        logger.error(f"Download failed: {e}")
        downloader._print_statistics()
    finally:
        db.close()


if __name__ == "__main__":
    main()