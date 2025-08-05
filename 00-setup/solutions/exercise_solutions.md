# Exercise Solutions - Module 00: Setup & Introduction

This document provides reference solutions and explanations for the Module 00 exercises.

## Exercise 1: Environment Check

### Expected Results

**Configuration Check:**

- Environment should be "development"
- Debug mode should be False (unless explicitly set)
- Default LLM provider should be configured
- Cache should be enabled

**API Keys Status:**

- At least one LLM provider should be configured (OpenAI, Anthropic, or Cohere)
- Optional services (Tavily, Langfuse) may or may not be configured

**Environment Validation:**

- Python version should be 3.11+
- All required packages should be installed
- LLM configuration should be available

### Common Issues & Solutions

1. **Missing API Keys**

   ```bash
   # Edit .env file and add:
   OPENAI_API_KEY=your_key_here
   # or
   ANTHROPIC_API_KEY=your_key_here
   # or  
   COHERE_API_KEY=your_key_here
   ```

2. **Package Import Errors**

   ```bash
   # Reinstall dependencies
   uv sync
   
   # Verify installation
   uv run verify_installation.py
   ```

3. **Configuration Problems**

   ```bash
   # Run setup script
   uv run 00-setup/setup_environment.py
   
   # Check .env file exists and has correct format
   ```

## Exercise 2: Your First DSPy Signature

### Reference Solution

Here's an example of a well-designed sentiment analysis signature:

```python
class SentimentAnalysis(dspy.Signature):
    """Analyze the sentiment of the given text and classify it as positive, negative, or neutral with a confidence score."""
    
    text = dspy.InputField(desc="The text to analyze for sentiment (e.g., reviews, comments, social media posts)")
    sentiment = dspy.OutputField(desc="The sentiment classification: 'positive', 'negative', or 'neutral'")
    confidence = dspy.OutputField(desc="Confidence score from 0.0 to 1.0, where 1.0 indicates highest confidence")
```

### Key Design Principles

1. **Clear Docstring**: Explains the task and expected outputs
2. **Specific Input Description**: Mentions types of text that work well
3. **Constrained Output**: Specifies exact format for sentiment
4. **Quantified Confidence**: Clear scale definition

### Alternative Approaches

**More Detailed Version:**

```python
class DetailedSentimentAnalysis(dspy.Signature):
    """Perform comprehensive sentiment analysis on text, providing classification, confidence, and reasoning."""
    
    text = dspy.InputField(desc="Text to analyze (reviews, social media, news, etc.)")
    sentiment = dspy.OutputField(desc="Sentiment: 'positive', 'negative', or 'neutral'")
    confidence = dspy.OutputField(desc="Confidence score (0.0-1.0)")
    reasoning = dspy.OutputField(desc="Brief explanation of the sentiment classification")
```

**Simplified Version:**

```python
class SimpleSentiment(dspy.Signature):
    """Classify text sentiment as positive, negative, or neutral."""
    
    text = dspy.InputField()
    sentiment = dspy.OutputField(desc="positive, negative, or neutral")
```

### Testing Strategy

**Good Test Cases:**

- Clear positive: "I love this product! It's amazing!"
- Clear negative: "This is terrible. Worst experience ever."
- Neutral: "The product works as expected."
- Mixed: "Good quality but expensive."
- Ambiguous: "It's okay, I guess."

**Expected Results:**

- Positive texts should return "positive" with high confidence
- Negative texts should return "negative" with high confidence  
- Neutral texts should return "neutral" with moderate confidence
- Mixed/ambiguous texts may have lower confidence scores

### Common Mistakes

1. **Vague Descriptions**

   ```python
   # Bad
   text = dspy.InputField(desc="input text")
   
   # Good  
   text = dspy.InputField(desc="The text to analyze for sentiment")
   ```

2. **Unclear Output Format**

   ```python
   # Bad
   sentiment = dspy.OutputField(desc="the sentiment")
   
   # Good
   sentiment = dspy.OutputField(desc="Sentiment: 'positive', 'negative', or 'neutral'")
   ```

3. **Missing Context**

   ```python
   # Bad
   class Sentiment(dspy.Signature):
       text = dspy.InputField()
       result = dspy.OutputField()
   
   # Good
   class SentimentAnalysis(dspy.Signature):
       """Analyze sentiment of text and classify as positive, negative, or neutral."""
       text = dspy.InputField(desc="Text to analyze for sentiment")
       sentiment = dspy.OutputField(desc="Sentiment classification: positive, negative, or neutral")
   ```

## Troubleshooting Guide

### Environment Issues

**Problem**: "Module not found" errors
**Solution**:

```bash
# Ensure you're in the project directory
cd DSPy-Zero-to-Expert

# Reinstall dependencies
uv sync

# Verify Python path
uv run python -c "import sys; print(sys.path)"
```

**Problem**: API key not recognized
**Solution**:

```bash
# Check .env file exists
ls -la .env

# Verify format (no spaces around =)
cat .env | grep API_KEY

# Restart notebook after changes
```

**Problem**: Marimo won't start
**Solution**:

```bash
# Check Marimo installation
uv run marimo --version

# Try running with explicit Python
uv run python -m marimo run notebook.py

# Check port availability
lsof -i :2718
```

### DSPy Issues

**Problem**: "No LM configured" error
**Solution**:

```python
from common import setup_dspy_environment
setup_dspy_environment()  # This should be called before using DSPy
```

**Problem**: API rate limiting
**Solution**:

- Wait a few minutes between requests
- Use smaller test cases
- Consider switching to a different provider

**Problem**: Unexpected results
**Solution**:

- Check signature descriptions are clear
- Try different temperature settings
- Add more specific constraints to output fields

## Next Steps

After completing these exercises, you should:

1. ✅ Have a working DSPy environment
2. ✅ Understand basic signature design
3. ✅ Know how to test signatures interactively
4. ✅ Be familiar with common troubleshooting steps

**Ready for Module 01?**

```bash
uv run marimo run 01-foundations/signatures_basics.py
```

## Additional Resources

- **DSPy Documentation**: <https://dspy.ai/>
- **Marimo Documentation**: <https://marimo.io/>
- **Project Troubleshooting**: `docs/TROUBLESHOOTING.md`
- **Best Practices**: `docs/BEST_PRACTICES.md`
