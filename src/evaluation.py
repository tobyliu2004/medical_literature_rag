"""
Evaluation Framework for Medical Literature RAG System.
Measures accuracy, relevance, citation quality, and response time on 10 sample questions.

Key Metrics:
1. Answer Accuracy - Is the answer medically correct?
2. Citation Quality - Are citations relevant and correct?
3. Retrieval Precision - Did we find the right papers?
4. Response Latency - How fast are we?
5. Confidence Calibration - Do confidence scores match accuracy?
"""

import json
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict

from src.rag_pipeline import RAGPipeline
from src.hybrid_search import HybridSearchEngine
from src.cache import CacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationQuestion:
    """Represents a test question with ground truth."""
    question: str
    expected_keywords: List[str]  # Keywords that should appear in answer
    expected_pmids: List[str] = field(default_factory=list)  # Papers that should be cited
    category: str = "general"  # Category for grouped analysis
    difficulty: str = "medium"  # easy, medium, hard
    min_confidence: float = 0.5  # Minimum acceptable confidence


@dataclass
class EvaluationResult:
    """Results for a single question evaluation."""
    question: str
    answer: str
    citations: List[Dict]
    confidence: float
    response_time: float
    
    # Metrics
    keyword_recall: float  # % of expected keywords found
    citation_precision: float  # % of citations that are relevant
    citation_recall: float  # % of expected citations found
    
    # Flags
    has_citations: bool
    confidence_calibrated: bool  # Is confidence aligned with accuracy?
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'question': self.question,
            'answer': self.answer[:200] + '...' if len(self.answer) > 200 else self.answer,
            'citations_count': len(self.citations),
            'confidence': round(self.confidence, 3),
            'response_time': round(self.response_time, 2),
            'metrics': {
                'keyword_recall': round(self.keyword_recall, 3),
                'citation_precision': round(self.citation_precision, 3),
                'citation_recall': round(self.citation_recall, 3)
            },
            'flags': {
                'has_citations': self.has_citations,
                'confidence_calibrated': self.confidence_calibrated
            }
        }


class EvaluationFramework:
    """
    Comprehensive evaluation system for the RAG pipeline.
    Tests accuracy, relevance, and performance.
    """
    
    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        """
        Initialize evaluation framework.
        
        Args:
            rag_pipeline: RAG pipeline to evaluate (creates new if None)
        """
        self.rag = rag_pipeline or RAGPipeline()
        self.searcher = HybridSearchEngine()
        self.cache = CacheManager()
        
        # Load evaluation questions
        self.questions = self._load_evaluation_questions()
        logger.info(f"Loaded {len(self.questions)} evaluation questions")
    
    def _load_evaluation_questions(self) -> List[EvaluationQuestion]:
        """
        Load or create evaluation questions.
        In production, load from a JSON file. For now, we'll create them.
        """
        questions = [
            # Cancer Immunotherapy Questions
            EvaluationQuestion(
                question="What are the main types of immune checkpoint inhibitors used in cancer treatment?",
                expected_keywords=["PD-1", "PD-L1", "CTLA-4", "pembrolizumab", "nivolumab", "ipilimumab"],
                category="immunotherapy",
                difficulty="easy"
            ),
            EvaluationQuestion(
                question="How does CAR-T cell therapy work for treating B-cell lymphomas?",
                expected_keywords=["chimeric antigen receptor", "T cells", "CD19", "B-cell", "lymphoma"],
                category="immunotherapy",
                difficulty="medium"
            ),
            EvaluationQuestion(
                question="What are the current limitations of immunotherapy in solid tumors?",
                expected_keywords=["tumor microenvironment", "immune suppression", "resistance", "penetration"],
                category="immunotherapy",
                difficulty="hard"
            ),
            
            # CRISPR/Gene Editing Questions
            EvaluationQuestion(
                question="What is CRISPR-Cas9 and how is it used in cancer research?",
                expected_keywords=["CRISPR", "Cas9", "gene editing", "guide RNA", "cancer"],
                category="gene_editing",
                difficulty="easy"
            ),
            EvaluationQuestion(
                question="What are the safety concerns with using CRISPR for cancer treatment?",
                expected_keywords=["off-target", "safety", "mutations", "ethics", "germline"],
                category="gene_editing",
                difficulty="medium"
            ),
            
            # AI/ML in Medicine Questions
            EvaluationQuestion(
                question="How is machine learning being used for cancer diagnosis?",
                expected_keywords=["machine learning", "artificial intelligence", "diagnosis", "imaging", "prediction"],
                category="ai_ml",
                difficulty="easy"
            ),
            EvaluationQuestion(
                question="What are the applications of deep learning in drug discovery?",
                expected_keywords=["deep learning", "drug discovery", "neural network", "screening", "prediction"],
                category="ai_ml",
                difficulty="medium"
            ),
            
            # Lung Cancer Specific
            EvaluationQuestion(
                question="What are the current first-line treatments for non-small cell lung cancer?",
                expected_keywords=["NSCLC", "chemotherapy", "immunotherapy", "targeted therapy", "EGFR", "ALK"],
                category="lung_cancer",
                difficulty="medium"
            ),
            EvaluationQuestion(
                question="How effective is immunotherapy for treating small cell lung cancer?",
                expected_keywords=["SCLC", "immunotherapy", "checkpoint", "survival", "response rate"],
                category="lung_cancer",
                difficulty="hard"
            ),
            
            # Combination Therapies
            EvaluationQuestion(
                question="What are the benefits of combining immunotherapy with chemotherapy?",
                expected_keywords=["combination", "synergy", "response rate", "survival", "toxicity"],
                category="combination",
                difficulty="medium",
                min_confidence=0.6
            )
        ]
        
        return questions
    
    def evaluate_question(self, 
                         eval_question: EvaluationQuestion,
                         use_cache: bool = False) -> EvaluationResult:
        """
        Evaluate a single question.
        
        Args:
            eval_question: Question to evaluate
            use_cache: Whether to use cached responses
            
        Returns:
            EvaluationResult with metrics
        """
        logger.info(f"Evaluating: {eval_question.question[:50]}...")
        
        # Clear cache if requested
        if not use_cache:
            self.cache.delete(f"rag_query_{eval_question.question}")
        
        # Generate answer
        start_time = time.time()
        response = self.rag.generate_answer(
            query=eval_question.question,
            max_papers=5,
            min_relevance=0.3
        )
        response_time = time.time() - start_time
        
        # Calculate keyword recall
        answer_lower = response.answer.lower()
        found_keywords = [kw for kw in eval_question.expected_keywords 
                         if kw.lower() in answer_lower]
        keyword_recall = len(found_keywords) / len(eval_question.expected_keywords) if eval_question.expected_keywords else 1.0
        
        # Calculate citation metrics
        cited_pmids = [c['pmid'] for c in response.citations]
        
        # Check if citations actually appear in the answer text
        citations_in_text = []
        for pmid in cited_pmids:
            if f"PMID: {pmid}" in response.answer or f"[PMID: {pmid}]" in response.answer:
                citations_in_text.append(pmid)
        
        # Citation precision: What % of citations are actually used in the answer?
        if cited_pmids:
            citation_precision = len(citations_in_text) / len(cited_pmids)
        else:
            citation_precision = 0.0
        
        # Citation recall (did we cite expected papers?)
        if eval_question.expected_pmids:
            found_pmids = [p for p in eval_question.expected_pmids if p in cited_pmids]
            citation_recall = len(found_pmids) / len(eval_question.expected_pmids)
        else:
            # If no expected PMIDs, check if we have ANY citations in the text
            citation_recall = 1.0 if citations_in_text else 0.0
        
        # Check confidence calibration
        accuracy_score = (keyword_recall + citation_precision + citation_recall) / 3
        confidence_calibrated = abs(response.confidence - accuracy_score) < 0.2
        
        return EvaluationResult(
            question=eval_question.question,
            answer=response.answer,
            citations=response.citations,
            confidence=response.confidence,
            response_time=response_time,
            keyword_recall=keyword_recall,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            has_citations=len(cited_pmids) > 0,
            confidence_calibrated=confidence_calibrated
        )
    
    def run_full_evaluation(self, 
                           categories: Optional[List[str]] = None,
                           max_questions: Optional[int] = None) -> Dict[str, Any]:
        """
        Run evaluation on all or selected questions.
        
        Args:
            categories: Specific categories to test (None = all)
            max_questions: Maximum questions to evaluate (None = all)
            
        Returns:
            Dictionary with comprehensive results
        """
        logger.info("="*60)
        logger.info("STARTING COMPREHENSIVE EVALUATION")
        logger.info("="*60)
        
        # Filter questions
        questions_to_eval = self.questions
        if categories:
            questions_to_eval = [q for q in questions_to_eval if q.category in categories]
        if max_questions:
            questions_to_eval = questions_to_eval[:max_questions]
        
        results = []
        category_metrics = defaultdict(list)
        
        for i, question in enumerate(questions_to_eval, 1):
            logger.info(f"\nEvaluating {i}/{len(questions_to_eval)}: {question.category}")
            
            try:
                result = self.evaluate_question(question, use_cache=False)
                results.append(result)
                
                # Group by category
                category_metrics[question.category].append({
                    'keyword_recall': result.keyword_recall,
                    'citation_precision': result.citation_precision,
                    'citation_recall': result.citation_recall,
                    'confidence': result.confidence,
                    'response_time': result.response_time
                })
                
                # Log progress
                logger.info(f"  ✓ Keyword Recall: {result.keyword_recall:.1%}")
                logger.info(f"  ✓ Citation Quality: {result.citation_precision:.1%}")
                logger.info(f"  ✓ Response Time: {result.response_time:.1f}s")
                
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")
                continue
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results, category_metrics)
        
        # Generate report
        report = self._generate_report(results, aggregate_metrics)
        
        return report
    
    def _calculate_aggregate_metrics(self, 
                                    results: List[EvaluationResult],
                                    category_metrics: Dict) -> Dict[str, Any]:
        """Calculate aggregate metrics across all evaluations."""
        
        if not results:
            return {}
        
        metrics = {
            'overall': {
                'total_questions': len(results),
                'avg_keyword_recall': np.mean([r.keyword_recall for r in results]),
                'avg_citation_precision': np.mean([r.citation_precision for r in results]),
                'avg_citation_recall': np.mean([r.citation_recall for r in results]),
                'avg_confidence': np.mean([r.confidence for r in results]),
                'avg_response_time': np.mean([r.response_time for r in results]),
                'median_response_time': np.median([r.response_time for r in results]),
                'pct_with_citations': sum(1 for r in results if r.has_citations) / len(results),
                'pct_calibrated': sum(1 for r in results if r.confidence_calibrated) / len(results)
            },
            'by_category': {}
        }
        
        # Category-specific metrics
        for category, cat_results in category_metrics.items():
            if cat_results:
                metrics['by_category'][category] = {
                    'count': len(cat_results),
                    'avg_keyword_recall': np.mean([r['keyword_recall'] for r in cat_results]),
                    'avg_citation_precision': np.mean([r['citation_precision'] for r in cat_results]),
                    'avg_response_time': np.mean([r['response_time'] for r in cat_results])
                }
        
        return metrics
    
    def _generate_report(self, 
                        results: List[EvaluationResult],
                        metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_questions': len(results),
                'rag_model': 'Llama-3.1-8B-Instruct',
                'embedding_model': 'all-mpnet-base-v2',
                'papers_in_db': 50007
            },
            'metrics': metrics,
            'results': [r.to_dict() for r in results],
            'summary': self._generate_summary(metrics)
        }
        
        return report
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable summary of results."""
        
        if not metrics or 'overall' not in metrics:
            return {'status': 'No results available'}
        
        overall = metrics['overall']
        
        # Determine overall performance level
        avg_accuracy = (overall['avg_keyword_recall'] + 
                       overall['avg_citation_precision'] + 
                       overall['avg_citation_recall']) / 3
        
        if avg_accuracy >= 0.8:
            performance = "EXCELLENT"
        elif avg_accuracy >= 0.6:
            performance = "GOOD"
        elif avg_accuracy >= 0.4:
            performance = "FAIR"
        else:
            performance = "NEEDS IMPROVEMENT"
        
        summary = {
            'performance_level': performance,
            'accuracy_score': f"{avg_accuracy:.1%}",
            'response_speed': f"{overall['avg_response_time']:.1f}s average",
            'citation_quality': f"{overall['avg_citation_precision']:.1%} precision",
            'confidence_calibration': f"{overall['pct_calibrated']:.1%} well-calibrated",
            'key_strength': self._identify_strength(metrics),
            'key_weakness': self._identify_weakness(metrics)
        }
        
        return summary
    
    def _identify_strength(self, metrics: Dict[str, Any]) -> str:
        """Identify the system's key strength."""
        overall = metrics['overall']
        
        strengths = []
        
        if overall['avg_response_time'] < 5:
            strengths.append("Fast response time")
        if overall['avg_citation_precision'] > 0.8:
            strengths.append("High citation quality")
        if overall['avg_keyword_recall'] > 0.7:
            strengths.append("Good content coverage")
        if overall['pct_calibrated'] > 0.7:
            strengths.append("Well-calibrated confidence")
        
        return strengths[0] if strengths else "Consistent performance"
    
    def _identify_weakness(self, metrics: Dict[str, Any]) -> str:
        """Identify the system's key weakness."""
        overall = metrics['overall']
        
        weaknesses = []
        
        if overall['avg_response_time'] > 10:
            weaknesses.append("Slow response time")
        if overall['avg_citation_recall'] < 0.5:
            weaknesses.append("Missing expected citations")
        if overall['avg_keyword_recall'] < 0.5:
            weaknesses.append("Incomplete answers")
        if overall['pct_with_citations'] < 0.8:
            weaknesses.append("Insufficient citations")
        
        return weaknesses[0] if weaknesses else "None identified"
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """
        Save evaluation report to JSON file.
        
        Args:
            report: Evaluation report dictionary
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.json"
        
        filepath = f"evaluations/{filename}"
        
        # Create directory if needed
        import os
        os.makedirs("evaluations", exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Report saved to {filepath}")
        return filepath


def run_evaluation():
    """Run a complete evaluation and save results."""
    
    print("="*60)
    print("MEDICAL LITERATURE RAG - EVALUATION FRAMEWORK")
    print("="*60)
    
    evaluator = EvaluationFramework()
    
    # Run evaluation on a subset for testing
    print("\nRunning evaluation on sample questions...")
    report = evaluator.run_full_evaluation(max_questions=5)
    
    # Display results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    summary = report['summary']
    metrics = report['metrics']['overall']
    
    print(f"\n📊 Performance Level: {summary['performance_level']}")
    print(f"📈 Accuracy Score: {summary['accuracy_score']}")
    print(f"⚡ Response Speed: {summary['response_speed']}")
    print(f"📚 Citation Quality: {summary['citation_quality']}")
    print(f"🎯 Confidence Calibration: {summary['confidence_calibration']}")
    
    print(f"\n✅ Key Strength: {summary['key_strength']}")
    print(f"⚠️  Key Weakness: {summary['key_weakness']}")
    
    print("\n📋 Detailed Metrics:")
    print(f"  - Keyword Recall: {metrics['avg_keyword_recall']:.1%}")
    print(f"  - Citation Precision: {metrics['avg_citation_precision']:.1%}")
    print(f"  - Citation Recall: {metrics['avg_citation_recall']:.1%}")
    print(f"  - Avg Response Time: {metrics['avg_response_time']:.1f}s")
    print(f"  - Papers with Citations: {metrics['pct_with_citations']:.1%}")
    
    # Save report
    filepath = evaluator.save_report(report)
    print(f"\n💾 Full report saved to: {filepath}")
    
    return report


if __name__ == "__main__":
    run_evaluation()