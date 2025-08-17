"""
Targeted Paper Acquisition System

This module fetches high-quality, clinically-relevant papers from PubMed
specifically for medical Q&A applications. Instead of random papers,
it focuses on FDA approvals, clinical trials, and treatment guidelines.
"""

import logging
import time
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

import requests
from tqdm import tqdm

from src.database_pool import DatabaseManager
from src.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

# PubMed API configuration
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_SEARCH_URL = f"{PUBMED_BASE_URL}/esearch.fcgi"
PUBMED_FETCH_URL = f"{PUBMED_BASE_URL}/efetch.fcgi"

# High-value search queries for medical Q&A
TARGETED_QUERIES = [
    # FDA approvals and drug information
    'FDA approved AND (cancer OR oncology)',
    'FDA approval AND clinical trial',
    'pembrolizumab OR nivolumab OR ipilimumab OR atezolizumab',
    'checkpoint inhibitor AND (mechanism OR target)',
    
    # Clinical guidelines
    'clinical guidelines AND treatment',
    'standard of care AND (cancer OR diabetes OR cardiovascular)',
    'first-line therapy OR second-line therapy',
    'treatment algorithm OR treatment protocol',
    
    # Drug mechanisms and targets
    'mechanism of action AND (drug OR therapy)',
    'molecular target AND therapeutic',
    'drug interaction AND clinical significance',
    'pharmacokinetics AND pharmacodynamics',
    
    # Clinical trial results
    'phase 3 clinical trial AND results',
    'randomized controlled trial AND efficacy',
    'clinical trial AND overall survival',
    'adverse events AND safety profile',
    
    # Specific diseases with treatments
    'melanoma AND immunotherapy',
    'non-small cell lung cancer AND targeted therapy',
    'breast cancer AND HER2',
    'diabetes AND GLP-1 agonist',
    'cardiovascular disease AND PCSK9 inhibitor'
]


@dataclass
class PaperQualityScore:
    """Score a paper's relevance for medical Q&A."""
    pmid: str
    title: str
    clinical_relevance: float  # 0-1
    specificity: float  # 0-1  
    recency: float  # 0-1
    authority: float  # 0-1 (based on journal impact)
    overall_score: float  # 0-1
    
    def meets_threshold(self, min_score: float = 0.7) -> bool:
        """Check if paper meets quality threshold."""
        return self.overall_score >= min_score


class TargetedAcquisition:
    """Fetch high-quality papers for medical Q&A."""
    
    def __init__(self):
        """Initialize the acquisition system."""
        self.db = DatabaseManager()
        self.embedding_gen = EmbeddingGenerator()
        self.existing_pmids = self._get_existing_pmids()
        
    def _get_existing_pmids(self) -> Set[str]:
        """Get PMIDs already in database."""
        with self.db.get_cursor() as cur:
            cur.execute("SELECT pmid FROM papers")
            return {row['pmid'] for row in cur.fetchall()}
    
    def search_pubmed(self, query: str, max_results: int = 100) -> List[str]:
        """
        Search PubMed for papers matching query.
        
        Args:
            query: Search query
            max_results: Maximum papers to retrieve
            
        Returns:
            List of PMIDs
        """
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance',
            'datetype': 'pdat',
            'mindate': '2018',  # Focus on recent papers
            'maxdate': '2024'
        }
        
        try:
            response = requests.get(PUBMED_SEARCH_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            pmids = data.get('esearchresult', {}).get('idlist', [])
            logger.info(f"Found {len(pmids)} papers for query: {query}")
            return pmids
            
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []
    
    def fetch_paper_details(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch full details for papers.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of paper dictionaries
        """
        if not pmids:
            return []
        
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml'
        }
        
        try:
            response = requests.get(PUBMED_FETCH_URL, params=params, timeout=60)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            papers = []
            
            for article in root.findall('.//PubmedArticle'):
                paper = self._parse_article(article)
                if paper and paper['abstract']:  # Only keep papers with abstracts
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"Failed to fetch paper details: {e}")
            return []
    
    def _parse_article(self, article_elem) -> Optional[Dict]:
        """Parse XML article element into dictionary."""
        try:
            # Extract PMID
            pmid = article_elem.find('.//PMID').text
            
            # Skip if already in database
            if pmid in self.existing_pmids:
                return None
            
            # Extract title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""
            
            # Extract abstract
            abstract_parts = []
            for abstract_text in article_elem.findall('.//AbstractText'):
                text = abstract_text.text or ""
                label = abstract_text.get('Label', '')
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)
            
            # Extract journal
            journal_elem = article_elem.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Extract publication date
            pub_date_elem = article_elem.find('.//PubDate')
            if pub_date_elem is not None:
                year = pub_date_elem.find('Year')
                month = pub_date_elem.find('Month')
                day = pub_date_elem.find('Day')
                
                if year is not None:
                    date_str = year.text
                    if month is not None:
                        date_str += f"-{month.text:0>2}"
                        if day is not None:
                            date_str += f"-{day.text:0>2}"
                    pub_date = date_str
                else:
                    pub_date = None
            else:
                pub_date = None
            
            # Extract authors
            authors = []
            for author in article_elem.findall('.//Author'):
                last_name = author.find('LastName')
                fore_name = author.find('ForeName')
                if last_name is not None:
                    name = last_name.text
                    if fore_name is not None:
                        name = f"{fore_name.text} {name}"
                    authors.append(name)
            
            return {
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'journal': journal,
                'pub_date': pub_date,
                'authors': authors
            }
            
        except Exception as e:
            logger.error(f"Failed to parse article: {e}")
            return None
    
    def score_paper_quality(self, paper: Dict) -> PaperQualityScore:
        """
        Score a paper's quality for medical Q&A.
        
        Scoring criteria:
        - Clinical relevance: Contains clinical terms, drug names, trial info
        - Specificity: Specific treatments, not general reviews
        - Recency: Newer papers score higher
        - Authority: High-impact journals score higher
        """
        pmid = paper['pmid']
        title = paper['title']
        abstract = paper['abstract'].lower()
        journal = paper.get('journal', '').lower()
        pub_date = paper.get('pub_date')
        
        # Clinical relevance (0-1)
        clinical_terms = [
            'fda', 'approved', 'clinical trial', 'phase', 'efficacy',
            'safety', 'adverse', 'treatment', 'therapy', 'patient',
            'survival', 'response rate', 'progression-free'
        ]
        clinical_count = sum(1 for term in clinical_terms if term in abstract)
        clinical_relevance = min(clinical_count / 5, 1.0)  # Max at 5 terms
        
        # Specificity (0-1)
        drug_names = [
            'pembrolizumab', 'nivolumab', 'ipilimumab', 'atezolizumab',
            'bevacizumab', 'trastuzumab', 'rituximab', 'cetuximab'
        ]
        has_drug_names = any(drug in abstract for drug in drug_names)
        has_numbers = any(char.isdigit() for char in abstract)  # Stats, doses
        specificity = 0.5
        if has_drug_names:
            specificity += 0.3
        if has_numbers:
            specificity += 0.2
        
        # Recency (0-1)
        recency = 0.5  # Default
        if pub_date:
            try:
                year = int(pub_date[:4])
                current_year = datetime.now().year
                years_old = current_year - year
                recency = max(0, 1 - (years_old / 10))  # Linear decay over 10 years
            except:
                pass
        
        # Authority (0-1)
        high_impact_journals = [
            'new england journal', 'nejm', 'lancet', 'jama', 
            'nature', 'science', 'cell', 'cancer cell',
            'journal of clinical oncology', 'annals of oncology'
        ]
        authority = 0.5  # Default
        for hij in high_impact_journals:
            if hij in journal:
                authority = 0.9
                break
        
        # Overall score (weighted average)
        overall_score = (
            clinical_relevance * 0.4 +
            specificity * 0.3 +
            recency * 0.2 +
            authority * 0.1
        )
        
        return PaperQualityScore(
            pmid=pmid,
            title=title,
            clinical_relevance=clinical_relevance,
            specificity=specificity,
            recency=recency,
            authority=authority,
            overall_score=overall_score
        )
    
    def acquire_targeted_papers(self, 
                               min_quality_score: float = 0.7,
                               papers_per_query: int = 50) -> int:
        """
        Main method to acquire high-quality papers.
        
        Args:
            min_quality_score: Minimum quality score to include paper
            papers_per_query: Papers to fetch per search query
            
        Returns:
            Number of papers added
        """
        logger.info("Starting targeted paper acquisition...")
        total_added = 0
        
        for query in tqdm(TARGETED_QUERIES, desc="Processing queries"):
            # Search PubMed
            pmids = self.search_pubmed(query, papers_per_query)
            
            # Filter out existing papers
            new_pmids = [p for p in pmids if p not in self.existing_pmids]
            
            if not new_pmids:
                continue
            
            # Fetch paper details
            papers = self.fetch_paper_details(new_pmids)
            
            # Score and filter papers
            high_quality_papers = []
            for paper in papers:
                score = self.score_paper_quality(paper)
                if score.meets_threshold(min_quality_score):
                    paper['quality_score'] = score.overall_score
                    high_quality_papers.append(paper)
                    logger.debug(f"High-quality paper: {paper['title'][:50]}... (score: {score.overall_score:.2f})")
            
            # Add to database
            if high_quality_papers:
                added = self._add_papers_to_db(high_quality_papers)
                total_added += added
                logger.info(f"Added {added} high-quality papers for query: {query}")
            
            # Rate limiting
            time.sleep(0.5)
        
        logger.info(f"✅ Acquisition complete. Added {total_added} high-quality papers.")
        return total_added
    
    def _add_papers_to_db(self, papers: List[Dict]) -> int:
        """Add papers to database with embeddings."""
        added_count = 0
        
        with self.db.get_cursor() as cur:
            for paper in papers:
                try:
                    # Generate embedding
                    text_for_embedding = f"{paper['title']} {paper['abstract']}"
                    embedding = self.embedding_gen.generate_embedding(text_for_embedding)
                    
                    # Insert paper
                    cur.execute("""
                        INSERT INTO papers (pmid, title, abstract, journal, pub_date, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (pmid) DO NOTHING
                    """, (
                        paper['pmid'],
                        paper['title'],
                        paper['abstract'],
                        paper.get('journal'),
                        paper.get('pub_date'),
                        embedding
                    ))
                    
                    if cur.rowcount > 0:
                        added_count += 1
                        self.existing_pmids.add(paper['pmid'])
                    
                except Exception as e:
                    logger.error(f"Failed to add paper {paper['pmid']}: {e}")
                    continue
            
            # Commit after batch
            cur.connection.commit()
        
        return added_count
    
    def analyze_improvement(self) -> Dict:
        """Analyze database improvement after acquisition."""
        with self.db.get_cursor() as cur:
            # Check new FDA/drug coverage
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN abstract ILIKE '%FDA%' THEN 1 ELSE 0 END) as fda_papers,
                    SUM(CASE WHEN abstract ILIKE '%pembrolizumab%' 
                              OR abstract ILIKE '%nivolumab%' THEN 1 ELSE 0 END) as drug_papers,
                    SUM(CASE WHEN abstract ILIKE '%clinical trial%' THEN 1 ELSE 0 END) as trial_papers
                FROM papers
            """)
            stats = cur.fetchone()
            
            return {
                'total_papers': stats['total'],
                'fda_coverage': stats['fda_papers'] / stats['total'] * 100,
                'drug_coverage': stats['drug_papers'] / stats['total'] * 100,
                'trial_coverage': stats['trial_papers'] / stats['total'] * 100
            }
    
    def close(self):
        """Clean up resources."""
        self.db.close()
        self.embedding_gen.close()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run targeted acquisition
    acquisition = TargetedAcquisition()
    
    try:
        # Acquire high-quality papers
        papers_added = acquisition.acquire_targeted_papers(
            min_quality_score=0.7,
            papers_per_query=25
        )
        
        # Analyze improvement
        stats = acquisition.analyze_improvement()
        
        print("\n📊 DATABASE IMPROVEMENT REPORT:")
        print("="*60)
        print(f"Total papers: {stats['total_papers']:,}")
        print(f"FDA coverage: {stats['fda_coverage']:.1f}% (target: >25%)")
        print(f"Drug name coverage: {stats['drug_coverage']:.1f}% (target: >20%)")
        print(f"Clinical trial coverage: {stats['trial_coverage']:.1f}% (target: >30%)")
        
    finally:
        acquisition.close()