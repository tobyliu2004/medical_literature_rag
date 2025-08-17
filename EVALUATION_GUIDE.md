# 📊 Evaluation Framework Guide

## Overview
Comprehensive evaluation system for the Medical Literature RAG system that measures accuracy, relevance, citation quality, and performance.

## Components

### 1. `src/evaluation.py` - Core Evaluation Framework
- **Purpose**: Measures system accuracy and performance
- **Key Features**:
  - 10 pre-defined medical questions across categories
  - Metrics: keyword recall, citation precision/recall, confidence calibration
  - Automated evaluation pipeline
  - JSON report generation

### 2. `src/benchmark.py` - Configuration Comparison
- **Purpose**: Tests different RAG configurations to find optimal settings
- **Configurations Tested**:
  - `baseline`: 70% vector, 30% keyword (default)
  - `vector_heavy`: 90% vector, 10% keyword
  - `keyword_heavy`: 30% vector, 70% keyword
  - `more_context`: Uses 10 papers instead of 5
  - `high_quality`: Fewer papers with higher relevance threshold
  - `balanced`: 50/50 vector and keyword

### 3. Metrics Measured

#### Quality Metrics
- **Keyword Recall**: % of expected medical terms found in answer
- **Citation Precision**: % of citations that are relevant
- **Citation Recall**: % of expected papers cited
- **Confidence Calibration**: Does confidence match accuracy?

#### Performance Metrics
- **Response Time**: Time to generate answer
- **Cache Performance**: Speed improvement with caching
- **Throughput**: Queries per second capability

## Running Evaluations

### Quick Test (2 questions, ~1 minute)
```bash
python test_evaluation.py
```

### Full Evaluation (10 questions, ~5 minutes)
```python
from src.evaluation import EvaluationFramework

evaluator = EvaluationFramework()
report = evaluator.run_full_evaluation()
evaluator.save_report(report)
```

### Benchmark Comparison (6 configs, ~10 minutes)
```python
from src.benchmark import BenchmarkSuite

suite = BenchmarkSuite()
report = suite.run_all_benchmarks()
suite.visualize_results(report)  # Creates charts
```

## Understanding Results

### Performance Levels
- **EXCELLENT**: >80% accuracy
- **GOOD**: 60-80% accuracy
- **FAIR**: 40-60% accuracy
- **NEEDS IMPROVEMENT**: <40% accuracy

### Key Metrics to Watch
1. **Keyword Recall**: Should be >70% for good coverage
2. **Citation Precision**: Should be >80% for reliability
3. **Response Time**: Target <5s for good UX
4. **Confidence Calibration**: Should be >60% for trustworthy scores

## Current Performance (50K Papers)

Based on testing with 50,007 papers:

### Strengths ✅
- Fast retrieval: 50-250ms search time
- Good keyword coverage: 75% recall
- Handles 50K papers efficiently
- Robust to edge cases (SQL injection, etc.)

### Areas for Improvement ⚠️
- Citation generation needs work (currently 0% - model not outputting PMIDs)
- Response time: 15-25s (can be improved with optimization)
- Confidence calibration could be better

## Optimization Strategies

### To Improve Quality:
1. Fine-tune prompt for better citation generation
2. Adjust retrieval parameters (vector vs keyword weights)
3. Increase min_relevance threshold
4. Use more papers for context (but slower)

### To Improve Speed:
1. Implement aggressive caching (already done with Redis)
2. Reduce number of papers retrieved
3. Use smaller model (trade-off with quality)
4. Pre-compute embeddings for common queries

## Interview Talking Points

When discussing this evaluation framework:

1. **Comprehensive Testing**: "I built a complete evaluation framework with 10 medical test questions across different categories"

2. **Metrics-Driven**: "I measure both quality (accuracy, citations) and performance (speed, throughput)"

3. **Optimization**: "I tested 6 different configurations to find the optimal balance between quality and speed"

4. **Visualization**: "The framework generates charts to visualize trade-offs between configurations"

5. **Production-Ready**: "It outputs JSON reports for tracking improvements over time"

6. **Real Issues Found**: "The evaluation revealed that citation generation needs improvement - this shows the framework actually works"

## Files Generated

- `evaluations/evaluation_report_*.json` - Detailed evaluation results
- `evaluations/benchmark_report_*.json` - Configuration comparison results
- `evaluations/benchmark_visualization_*.png` - Performance charts

## Next Steps

1. **Fix Citation Generation**: Modify prompt to ensure PMIDs are included
2. **Speed Optimization**: Implement query result caching
3. **Expand Test Set**: Add more evaluation questions
4. **A/B Testing**: Compare different models (Llama 3.1 vs others)
5. **User Studies**: Get feedback from medical professionals