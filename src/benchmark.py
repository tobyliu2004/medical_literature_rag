"""
Tests 6 different RAG configurations to find the optimal performance settings

Benchmarking system for comparing different RAG configurations.
Tests various settings to find optimal performance.
Perfect for demonstrating optimization skills in interviews.
"""

import json
import time
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.rag_pipeline import RAGPipeline
from src.hybrid_search import HybridSearchEngine
from src.evaluation import EvaluationFramework, EvaluationQuestion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark test."""
    name: str
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    max_papers: int = 5
    min_relevance: float = 0.3
    temperature: float = 0.3
    max_tokens: int = 512
    use_cache: bool = True
    description: str = ""


class BenchmarkSuite:
    """
    Comprehensive benchmarking system for RAG configurations.
    Tests different settings and finds optimal parameters.
    """
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.configs = self._create_benchmark_configs()
        self.evaluator = EvaluationFramework()
        
    def _create_benchmark_configs(self) -> List[BenchmarkConfig]:
        """Create different configurations to test."""
        
        configs = [
            # Baseline configuration
            BenchmarkConfig(
                name="baseline",
                vector_weight=0.7,
                keyword_weight=0.3,
                max_papers=5,
                min_relevance=0.3,
                description="Default balanced configuration"
            ),
            
            # Vector-heavy search
            BenchmarkConfig(
                name="vector_heavy",
                vector_weight=0.9,
                keyword_weight=0.1,
                max_papers=5,
                min_relevance=0.3,
                description="Emphasizes semantic similarity"
            ),
            
            # Keyword-heavy search
            BenchmarkConfig(
                name="keyword_heavy",
                vector_weight=0.3,
                keyword_weight=0.7,
                max_papers=5,
                min_relevance=0.3,
                description="Emphasizes exact term matching"
            ),
            
            # More papers
            BenchmarkConfig(
                name="more_context",
                vector_weight=0.7,
                keyword_weight=0.3,
                max_papers=10,
                min_relevance=0.2,
                description="Uses more papers for context"
            ),
            
            # Fewer papers, higher quality
            BenchmarkConfig(
                name="high_quality",
                vector_weight=0.7,
                keyword_weight=0.3,
                max_papers=3,
                min_relevance=0.5,
                description="Fewer but more relevant papers"
            ),
            
            # Balanced 50/50
            BenchmarkConfig(
                name="balanced",
                vector_weight=0.5,
                keyword_weight=0.5,
                max_papers=5,
                min_relevance=0.3,
                description="Equal weight to vector and keyword"
            )
        ]
        
        return configs
    
    def benchmark_config(self, 
                        config: BenchmarkConfig,
                        questions: List[EvaluationQuestion]) -> Dict[str, Any]:
        """
        Benchmark a single configuration.
        
        Args:
            config: Configuration to test
            questions: Questions to evaluate
            
        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Benchmarking: {config.name}")
        
        # Temporarily modify search engine settings
        original_searcher = self.evaluator.searcher
        self.evaluator.searcher = HybridSearchEngine()
        
        results = []
        response_times = []
        
        for question in questions:
            start_time = time.time()
            
            # Override RAG parameters for this test
            response = self.evaluator.rag.generate_answer(
                query=question.question,
                max_papers=config.max_papers,
                min_relevance=config.min_relevance
            )
            
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            # Evaluate quality
            eval_result = self.evaluator.evaluate_question(question, use_cache=False)
            results.append(eval_result)
        
        # Restore original searcher
        self.evaluator.searcher = original_searcher
        
        # Calculate metrics
        metrics = {
            'config_name': config.name,
            'description': config.description,
            'parameters': {
                'vector_weight': config.vector_weight,
                'keyword_weight': config.keyword_weight,
                'max_papers': config.max_papers,
                'min_relevance': config.min_relevance
            },
            'performance': {
                'avg_keyword_recall': np.mean([r.keyword_recall for r in results]),
                'avg_citation_precision': np.mean([r.citation_precision for r in results]),
                'avg_citation_recall': np.mean([r.citation_recall for r in results]),
                'avg_confidence': np.mean([r.confidence for r in results]),
                'avg_response_time': np.mean(response_times),
                'std_response_time': np.std(response_times),
                'min_response_time': np.min(response_times),
                'max_response_time': np.max(response_times)
            },
            'quality_score': self._calculate_quality_score(results),
            'speed_score': self._calculate_speed_score(response_times)
        }
        
        return metrics
    
    def _calculate_quality_score(self, results: List) -> float:
        """
        Calculate overall quality score (0-100).
        
        Weighted combination of:
        - 40% keyword recall
        - 30% citation precision
        - 20% citation recall
        - 10% confidence calibration
        """
        if not results:
            return 0.0
        
        keyword_score = np.mean([r.keyword_recall for r in results]) * 40
        precision_score = np.mean([r.citation_precision for r in results]) * 30
        recall_score = np.mean([r.citation_recall for r in results]) * 20
        calibration_score = sum(1 for r in results if r.confidence_calibrated) / len(results) * 10
        
        return keyword_score + precision_score + recall_score + calibration_score
    
    def _calculate_speed_score(self, response_times: List[float]) -> float:
        """
        Calculate speed score (0-100).
        
        Based on:
        - <2s = 100
        - 2-5s = 80
        - 5-10s = 60
        - 10-15s = 40
        - >15s = 20
        """
        avg_time = np.mean(response_times)
        
        if avg_time < 2:
            return 100
        elif avg_time < 5:
            return 80
        elif avg_time < 10:
            return 60
        elif avg_time < 15:
            return 40
        else:
            return 20
    
    def run_all_benchmarks(self, 
                          sample_size: int = 5) -> Dict[str, Any]:
        """
        Run all benchmark configurations.
        
        Args:
            sample_size: Number of questions to test per config
            
        Returns:
            Comprehensive benchmark results
        """
        logger.info("="*60)
        logger.info("RUNNING BENCHMARK SUITE")
        logger.info("="*60)
        
        # Get sample questions
        questions = self.evaluator.questions[:sample_size]
        
        all_results = []
        
        for config in self.configs:
            logger.info(f"\nTesting: {config.name}")
            result = self.benchmark_config(config, questions)
            all_results.append(result)
            
            # Display progress
            logger.info(f"  Quality Score: {result['quality_score']:.1f}/100")
            logger.info(f"  Speed Score: {result['speed_score']:.1f}/100")
            logger.info(f"  Avg Response: {result['performance']['avg_response_time']:.1f}s")
        
        # Find best configuration
        best_quality = max(all_results, key=lambda x: x['quality_score'])
        best_speed = min(all_results, key=lambda x: x['performance']['avg_response_time'])
        best_overall = max(all_results, 
                          key=lambda x: x['quality_score'] * 0.7 + x['speed_score'] * 0.3)
        
        benchmark_report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configurations_tested': len(self.configs),
            'questions_per_config': sample_size,
            'results': all_results,
            'best_configurations': {
                'quality': best_quality['config_name'],
                'speed': best_speed['config_name'],
                'overall': best_overall['config_name']
            },
            'recommendations': self._generate_recommendations(all_results)
        }
        
        return benchmark_report
    
    def _generate_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on benchmarks."""
        
        recommendations = []
        
        # Find average metrics
        avg_quality = np.mean([r['quality_score'] for r in results])
        avg_speed = np.mean([r['performance']['avg_response_time'] for r in results])
        
        # Quality recommendations
        if avg_quality < 60:
            recommendations.append("Consider fine-tuning the model on medical data")
            recommendations.append("Increase the number of papers in the database")
        elif avg_quality < 80:
            recommendations.append("Optimize retrieval parameters for better relevance")
        
        # Speed recommendations
        if avg_speed > 10:
            recommendations.append("Implement more aggressive caching strategies")
            recommendations.append("Consider using a smaller model for faster inference")
        elif avg_speed > 5:
            recommendations.append("Pre-compute embeddings for common queries")
        
        # Config-specific recommendations
        best_config = max(results, key=lambda x: x['quality_score'])
        if best_config['parameters']['vector_weight'] > 0.7:
            recommendations.append("Vector search performs best - ensure embeddings are high quality")
        elif best_config['parameters']['keyword_weight'] > 0.5:
            recommendations.append("Keyword search is important - maintain good text indexes")
        
        return recommendations
    
    def visualize_results(self, benchmark_report: Dict[str, Any]):
        """
        Create visualizations of benchmark results.
        
        Args:
            benchmark_report: Results from run_all_benchmarks
        """
        results = benchmark_report['results']
        
        # Set up the plot style
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Quality vs Speed scatter plot
        ax1 = axes[0, 0]
        quality_scores = [r['quality_score'] for r in results]
        response_times = [r['performance']['avg_response_time'] for r in results]
        config_names = [r['config_name'] for r in results]
        
        ax1.scatter(response_times, quality_scores, s=100, alpha=0.7)
        for i, name in enumerate(config_names):
            ax1.annotate(name, (response_times[i], quality_scores[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax1.set_xlabel('Avg Response Time (s)')
        ax1.set_ylabel('Quality Score (0-100)')
        ax1.set_title('Quality vs Speed Trade-off')
        ax1.grid(True, alpha=0.3)
        
        # 2. Configuration comparison bar chart
        ax2 = axes[0, 1]
        x = np.arange(len(config_names))
        width = 0.35
        
        ax2.bar(x - width/2, quality_scores, width, label='Quality Score', alpha=0.8)
        ax2.bar(x + width/2, [r['speed_score'] for r in results], width, 
                label='Speed Score', alpha=0.8)
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Score (0-100)')
        ax2.set_title('Configuration Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(config_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Detailed metrics heatmap
        ax3 = axes[1, 0]
        metrics_data = []
        for r in results:
            metrics_data.append([
                r['performance']['avg_keyword_recall'],
                r['performance']['avg_citation_precision'],
                r['performance']['avg_citation_recall'],
                r['performance']['avg_confidence']
            ])
        
        im = ax3.imshow(metrics_data, aspect='auto', cmap='YlOrRd')
        ax3.set_xticks(np.arange(4))
        ax3.set_yticks(np.arange(len(config_names)))
        ax3.set_xticklabels(['Keyword\nRecall', 'Citation\nPrecision', 
                            'Citation\nRecall', 'Confidence'])
        ax3.set_yticklabels(config_names)
        ax3.set_title('Detailed Metrics Heatmap')
        
        # Add colorbar
        plt.colorbar(im, ax=ax3)
        
        # 4. Response time distribution
        ax4 = axes[1, 1]
        response_data = [r['performance']['avg_response_time'] for r in results]
        error_data = [r['performance']['std_response_time'] for r in results]
        
        ax4.barh(config_names, response_data, xerr=error_data, alpha=0.7)
        ax4.set_xlabel('Response Time (s)')
        ax4.set_ylabel('Configuration')
        ax4.set_title('Response Time Comparison')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('RAG System Benchmark Results', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save the plot
        import os
        os.makedirs('evaluations', exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f'evaluations/benchmark_visualization_{timestamp}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Visualization saved to {filename}")
        
        return fig
    
    def save_benchmark_report(self, report: Dict[str, Any], filename: str = None):
        """Save benchmark report to JSON file."""
        
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"benchmark_report_{timestamp}.json"
        
        import os
        os.makedirs('evaluations', exist_ok=True)
        filepath = f"evaluations/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Benchmark report saved to {filepath}")
        return filepath


def run_benchmarks():
    """Run complete benchmark suite."""
    
    print("="*60)
    print("RAG SYSTEM BENCHMARK SUITE")
    print("="*60)
    
    suite = BenchmarkSuite()
    
    # Run benchmarks
    print("\nRunning benchmark configurations...")
    report = suite.run_all_benchmarks(sample_size=3)
    
    # Display results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    print(f"\n📊 Configurations Tested: {report['configurations_tested']}")
    print(f"📋 Questions per Config: {report['questions_per_config']}")
    
    print("\n🏆 Best Configurations:")
    print(f"  Quality: {report['best_configurations']['quality']}")
    print(f"  Speed: {report['best_configurations']['speed']}")
    print(f"  Overall: {report['best_configurations']['overall']}")
    
    print("\n💡 Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Save report
    filepath = suite.save_benchmark_report(report)
    print(f"\n💾 Report saved to: {filepath}")
    
    # Create visualizations
    print("\n📊 Generating visualizations...")
    suite.visualize_results(report)
    
    return report


if __name__ == "__main__":
    run_benchmarks()