#!/usr/bin/env python3
"""
Quick test script to demonstrate the evaluation framework.
Shows how the system performs on medical questions.
"""

import json
from src.evaluation import EvaluationFramework
from src.benchmark import BenchmarkSuite

def main():
    print("="*60)
    print("MEDICAL LITERATURE RAG - EVALUATION DEMO")
    print("="*60)
    
    # Test 1: Quick Evaluation
    print("\n📊 Running Quick Evaluation (2 questions)...")
    print("-"*40)
    
    evaluator = EvaluationFramework()
    report = evaluator.run_full_evaluation(max_questions=2)
    
    # Display results
    summary = report['summary']
    metrics = report['metrics']['overall']
    
    print(f"\nPerformance Level: {summary['performance_level']}")
    print(f"Accuracy Score: {summary['accuracy_score']}")
    print(f"Response Speed: {summary['response_speed']}")
    print(f"Key Strength: {summary['key_strength']}")
    print(f"Key Weakness: {summary['key_weakness']}")
    
    print("\nDetailed Metrics:")
    print(f"  - Keyword Recall: {metrics['avg_keyword_recall']:.1%}")
    print(f"  - Citation Precision: {metrics['avg_citation_precision']:.1%}")
    print(f"  - Avg Response Time: {metrics['avg_response_time']:.1f}s")
    print(f"  - With Citations: {metrics['pct_with_citations']:.1%}")
    
    # Save report
    filepath = evaluator.save_report(report)
    print(f"\n💾 Evaluation report saved to: {filepath}")
    
    # Test 2: Quick Benchmark (optional - takes longer)
    user_input = input("\n🔬 Run benchmark comparison? (y/n): ")
    if user_input.lower() == 'y':
        print("\n📊 Running Benchmark Comparison (this may take a minute)...")
        print("-"*40)
        
        suite = BenchmarkSuite()
        benchmark_report = suite.run_all_benchmarks(sample_size=2)
        
        print(f"\n🏆 Best Configurations:")
        print(f"  Quality: {benchmark_report['best_configurations']['quality']}")
        print(f"  Speed: {benchmark_report['best_configurations']['speed']}")
        print(f"  Overall: {benchmark_report['best_configurations']['overall']}")
        
        # Save benchmark report
        benchmark_path = suite.save_benchmark_report(benchmark_report)
        print(f"\n💾 Benchmark report saved to: {benchmark_path}")
        
        # Generate visualizations
        print("📊 Generating visualizations...")
        suite.visualize_results(benchmark_report)
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    print("\n🎯 What This Shows:")
    print("  1. System can answer medical questions")
    print("  2. Retrieves relevant papers from 50K database")
    print("  3. Measures accuracy and performance")
    print("  4. Identifies strengths and weaknesses")
    print("  5. Compares different configurations")
    
    print("\n💡 For Interviews:")
    print("  - Show the evaluation metrics")
    print("  - Explain the trade-offs (quality vs speed)")
    print("  - Discuss optimization strategies")
    print("  - Demonstrate continuous improvement")

if __name__ == "__main__":
    main()