"""
PubMed Data Fetcher - Our First Script
This fetches cancer research papers from PubMed's E-utilities API.

Key concepts we're learning:
1. API interaction with rate limiting
2. Error handling and retries
3. Data validation
4. JSON serialization
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration from environment
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")

# PubMed API endpoints
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SEARCH_URL = f"{BASE_URL}/esearch.fcgi"
FETCH_URL = f"{BASE_URL}/efetch.fcgi"

# Rate limiting
# With API key: 10 requests/second, without: 3 requests/second
RATE_LIMIT_DELAY = 0.1 if PUBMED_API_KEY else 0.34  # seconds between requests


class PubMedFetcher:
    """Handles all interactions with PubMed E-utilities API"""
    
    def __init__(self):
        """Initialize the fetcher with configuration"""
        if not PUBMED_EMAIL:
            raise ValueError(
                "PUBMED_EMAIL is required! Please add your email to .env file.\n"
                "PubMed requires an email for their API terms of service."
            )
        
        self.session = requests.Session()
        self.request_count = 0
        
        # Base parameters for all requests
        self.base_params = {
            "email": PUBMED_EMAIL,
            "tool": "medical_rag_system"
        }
        
        if PUBMED_API_KEY:
            self.base_params["api_key"] = PUBMED_API_KEY
            print(f"✓ Using API key (10 req/sec limit)")
        else:
            print(f"! No API key found (3 req/sec limit)")
    
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        time.sleep(RATE_LIMIT_DELAY)
        self.request_count += 1
    
    def search(self, query: str, max_results: int = 10) -> List[str]:
        """
        Search PubMed for papers matching the query.
        
        Args:
            query: Search terms (e.g., "lung cancer immunotherapy")
            max_results: Maximum number of results to return
            
        Returns:
            List of PubMed IDs (PMIDs)
        """
        print(f"\n🔍 Searching PubMed for: '{query}'")
        print(f"   Max results: {max_results}")
        
        params = {
            **self.base_params,
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",  # Most relevant first
            "datetype": "pdat",   # Publication date
            "mindate": "2020",    # Papers from 2020 onwards
            "maxdate": "2024"
        }
        
        self._rate_limit()
        
        try:
            response = self.session.get(SEARCH_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            pmids = data["esearchresult"]["idlist"]
            
            print(f"✓ Found {len(pmids)} papers")
            return pmids
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error searching PubMed: {e}")
            return []
    
    def fetch_papers(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch full paper details for given PubMed IDs.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of paper dictionaries with all metadata
        """
        if not pmids:
            return []
        
        print(f"\n📥 Fetching details for {len(pmids)} papers...")
        
        # PubMed allows fetching multiple papers at once
        params = {
            **self.base_params,
            "db": "pubmed",
            "id": ",".join(pmids),  # Comma-separated IDs
            "retmode": "xml",
            "rettype": "abstract"
        }
        
        self._rate_limit()
        
        try:
            response = self.session.get(FETCH_URL, params=params)
            response.raise_for_status()
            
            # Parse XML response
            papers = self._parse_xml(response.text)
            print(f"✓ Successfully fetched {len(papers)} papers")
            
            return papers
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching papers: {e}")
            return []
    
    def _parse_xml(self, xml_text: str) -> List[Dict]:
        """
        Parse PubMed XML response into structured data.
        
        This is a simplified parser - in production, use lxml or xml.etree
        """
        # For now, we'll use a simple approach
        # In the next iteration, we'll use proper XML parsing
        import xml.etree.ElementTree as ET
        
        papers = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for article in root.findall(".//PubmedArticle"):
                paper = self._extract_paper_data(article)
                if paper:
                    papers.append(paper)
                    
        except ET.ParseError as e:
            print(f"✗ Error parsing XML: {e}")
        
        return papers
    
    def _extract_paper_data(self, article_elem) -> Optional[Dict]:
        """Extract paper data from XML element"""
        try:
            # Get PMID
            pmid = article_elem.find(".//PMID")
            if pmid is None:
                return None
            
            # Get article details
            article = article_elem.find(".//Article")
            if article is None:
                return None
            
            # Extract title
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "No title"
            
            # Extract abstract
            abstract_elem = article.find(".//AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            # Extract authors
            authors = []
            for author in article.findall(".//Author"):
                last_name = author.find(".//LastName")
                first_name = author.find(".//ForeName")
                if last_name is not None:
                    name = last_name.text
                    if first_name is not None:
                        name = f"{name}, {first_name.text}"
                    authors.append(name)
            
            # Extract journal
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Extract publication date
            pub_date_elem = article.find(".//PubDate")
            pub_year = pub_date_elem.find(".//Year")
            pub_date = pub_year.text if pub_year is not None else "Unknown"
            
            # Extract MeSH terms (medical keywords)
            mesh_terms = []
            for mesh in article_elem.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text)
            
            return {
                "pmid": pmid.text,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "journal": journal,
                "pub_date": pub_date,
                "mesh_terms": mesh_terms,
                "fetched_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"✗ Error extracting paper data: {e}")
            return None


def main():
    """Main function to demonstrate fetching papers"""
    
    print("=" * 60)
    print("Medical Literature RAG - PubMed Fetcher")
    print("=" * 60)
    
    # Initialize fetcher
    try:
        fetcher = PubMedFetcher()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        return
    
    # Search for cancer papers
    # We'll start with a specific query to get relevant results
    query = "lung cancer immunotherapy treatment 2023"
    
    # Search for papers
    pmids = fetcher.search(query, max_results=10)
    
    if not pmids:
        print("\n❌ No papers found!")
        return
    
    # Fetch full paper details
    papers = fetcher.fetch_papers(pmids)
    
    if not papers:
        print("\n❌ Failed to fetch paper details!")
        return
    
    # Save to JSON file in data directory
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    output_file = "data/cancer_papers_sample.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(papers)} papers to {output_file}")
    
    # Display summary
    print("\n" + "=" * 60)
    print("SUMMARY OF FETCHED PAPERS")
    print("=" * 60)
    
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['title'][:80]}...")
        print(f"   Authors: {', '.join(paper['authors'][:3])}")
        if len(paper['authors']) > 3:
            print(f"            ... and {len(paper['authors']) - 3} more")
        print(f"   Journal: {paper['journal']}")
        print(f"   Year: {paper['pub_date']}")
        print(f"   Abstract: {len(paper['abstract'])} characters")
        print(f"   MeSH terms: {len(paper['mesh_terms'])} terms")
    
    print("\n" + "=" * 60)
    print(f"Total API requests made: {fetcher.request_count}")
    print(f"Papers with abstracts: {sum(1 for p in papers if p['abstract'])}/{len(papers)}")
    print("=" * 60)


if __name__ == "__main__":
    main()