# Architecture Decisions

## Model Selection: Llama 3.1 8B Instruct vs BioMistral 7B

### Decision
Migrated from BioMistral 7B to Llama 3.1 8B Instruct (Q4_K_M quantization)

### Context
Our medical literature RAG system was experiencing critical issues:
- Citation accuracy: 50% (only "How does..." questions worked)
- Response time: 30s average
- Inconsistent behavior between question types

### Evaluation Results

#### Before (BioMistral 7B):
- Citation Precision: 50%
- Response Time: 30s average
- Issue: Model wasn't trained for citation generation
- Pattern: "What are..." questions → 0% citations
- Pattern: "How does..." questions → 100% citations

#### After (Llama 3.1 8B Instruct):
- Citation Precision: 100% ✅
- Response Time: 9.8s average ✅
- All question types now properly cite sources
- Consistent behavior across query patterns

### Rationale

1. **Instruction Following > Domain Knowledge**
   - BioMistral was trained on medical texts but not for following specific formatting instructions
   - Llama 3.1 was specifically trained for instruction following
   - We provide medical context via retrieved papers, so domain knowledge is less critical

2. **Production-Grade Thinking**
   - For medical applications, accuracy trumps speed
   - Researchers would wait for reliable citations
   - 100% citation accuracy is non-negotiable for credibility

3. **Technical Advantages**
   - Better Metal/GPU optimization (28 layers on GPU vs 10)
   - Larger context window (8K usable, 128K available)
   - More consistent output formatting

### Trade-offs

- **Storage**: 4.6GB (Llama) vs 4.1GB (BioMistral) - minimal difference
- **RAM Usage**: ~5-6GB when running - acceptable for production
- **Speed**: 3x faster (9.8s vs 30s) - actually improved!

### Implementation Details

```python
# Model configuration optimized for M3 Pro
MODEL_PATH = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
MODEL_PARAMS = {
    'n_ctx': 8192,         # Increased context window
    'n_threads': 8,        # M3 Pro optimization
    'n_gpu_layers': 28,    # Maximum GPU utilization
    'temperature': 0.1,    # Low for consistency
    'top_p': 0.95,
    'max_tokens': 512,
    'repeat_penalty': 1.1  # Prevent repetition
}
```

### Lessons Learned

1. **Always validate metrics** - Our initial evaluation was returning fake 100% precision
2. **Test with diverse queries** - Different question patterns revealed the citation bug
3. **Choose models based on task requirements** - Instruction following was more important than medical knowledge for our use case

### Future Considerations

- Could explore Llama 3.2 or newer models as they release
- Consider fine-tuning on medical citation formats if we had resources
- Monitor token usage costs if deploying to production API

### Interview Talking Points

"We discovered our evaluation framework was lying - it always returned 100% citation precision if any citations existed. After fixing it, we found BioMistral only achieved 50% accuracy. By analyzing the failure patterns, we realized the issue wasn't medical knowledge but instruction following. Switching to Llama 3.1 8B Instruct improved citation accuracy to 100% while actually reducing response time by 3x due to better GPU optimization."