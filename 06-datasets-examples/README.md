# Module 06: Datasets & Examples Management

This module focuses on comprehensive dataset management for DSPy applications, covering data loading, preprocessing, validation, and quality assessment. You'll learn how to work with DSPy Example objects effectively and build robust data pipelines through interactive Marimo notebooks.

## Learning Objectives

By the end of this module, you will be able to:

1. **Load and manage datasets** from various formats (JSON, CSV, JSONL)
2. **Create and validate DSPy Example objects** with proper error handling
3. **Implement data preprocessing pipelines** for cleaning and transformation
4. **Assess data quality** using comprehensive metrics and validation
5. **Split and sample datasets** for training, validation, and testing
6. **Build interactive data exploration tools** using Marimo

## 📚 Learning Path - Interactive Notebooks

Follow these Marimo notebooks in order for the best learning experience:

### 1. 📁 Dataset Loading Basics

**File:** `dataset_loading_basics.py`  
**Duration:** 30-45 minutes

**What you'll learn:**

- Load datasets from JSON, JSONL, and CSV formats
- Create and manipulate DSPy Example objects
- Validate loaded data with interactive tools
- Export datasets in different formats

**Key Features:**

- Interactive format selection
- Real-time data loading and validation
- Dynamic example creation forms
- Live data exploration with sliders

**Run with:**

```bash
uv run marimo run dataset_loading_basics.py
```

### 2. 🧹 Data Preprocessing Interactive

**File:** `data_preprocessing_interactive.py`  
**Duration:** 45-60 minutes

**What you'll learn:**

- Configure preprocessing pipelines with interactive controls
- Apply text cleaning and normalization
- Use quality filters to improve data
- Compare before/after preprocessing results

**Key Features:**

- Interactive preprocessing configuration
- Real-time before/after comparison
- Quality improvement metrics
- Export processed datasets

**Run with:**

```bash
uv run marimo run data_preprocessing_interactive.py
```

### 3. 📊 Data Quality Assessment (Coming Soon)

**File:** `data_quality_interactive.py`  
**Duration:** 30-45 minutes

**What you'll learn:**

- Comprehensive quality metrics calculation
- Bias detection and analysis
- Generate quality reports
- Set quality thresholds and alerts

### 4. 🔍 Data Exploration Dashboard (Coming Soon)

**File:** `data_exploration_dashboard.py`  
**Duration:** 45-60 minutes

**What you'll learn:**

- Interactive data visualization
- Pattern detection and analysis
- Statistical exploration tools
- Export insights and reports

## Module Structure

```
06-datasets-examples/
├── README.md                           # This file
├── dataset_loading_basics.py           # 📁 Interactive notebook: Dataset loading
├── data_preprocessing_interactive.py   # 🧹 Interactive notebook: Preprocessing
├── dataset_management.py               # Core dataset management utilities
├── data_preprocessing.py               # Data preprocessing pipeline
├── data_quality_metrics.py             # Quality assessment tools
├── data/                               # Sample datasets
│   ├── sample_qa.json                 # Question-answer dataset
│   ├── sample_classification.csv       # Classification dataset
│   └── sample_rag.jsonl               # RAG dataset
└── solutions/                         # Exercise solutions (Python scripts)
    ├── solution_01_basic_loading.py
    ├── solution_02_preprocessing.py
    ├── solution_03_quality_assessment.py
    └── solution_04_data_exploration.py
```

## Key Concepts

### DSPy Example Objects

DSPy uses `Example` objects to represent training and evaluation data:

```python
from dspy import Example

# Create a simple example
example = Example(question="What is Python?", answer="A programming language")

# Access fields
print(example.question)  # "What is Python?"
print(example.answer)    # "A programming language"

# Examples can have any fields
example = Example(
    text="This is a great product!",
    label="positive",
    confidence=0.95,
    metadata={"source": "review_site"}
)
```

### Dataset Management Workflow

1. **Loading**: Import data from various formats
2. **Validation**: Check for required fields and data quality
3. **Preprocessing**: Clean and transform data
4. **Splitting**: Create train/validation/test sets
5. **Quality Assessment**: Analyze data characteristics and issues

### Data Quality Metrics

- **Completeness**: Percentage of non-missing values
- **Consistency**: Data type and format consistency
- **Uniqueness**: Diversity of values in fields
- **Validity**: Adherence to expected formats and ranges

## Getting Started

### Quick Start

1. **Start with the first notebook**:

   ```bash
   marimo run dataset_loading_basics.py
   ```

2. **Follow the interactive guide** through each step

3. **Progress to preprocessing**:

   ```bash
   marimo run data_preprocessing_interactive.py
   ```

4. **Explore the solution scripts** for additional examples

### Prerequisites

- Completion of Modules 00-05
- Understanding of DSPy basics and optimization
- Familiarity with data manipulation concepts

## Common Use Cases

### Question-Answering Datasets

```python
# Load QA dataset
examples = manager.load_from_json("qa_dataset.json")

# Validate required fields
valid_examples, errors = manager.validate_examples(
    examples, 
    required_fields=["question", "answer"]
)

# Split for training
splits = manager.split_dataset(valid_examples)
```

### Classification Datasets

```python
# Load from CSV
examples = manager.load_from_csv(
    "classification.csv",
    input_col="text",
    output_col="label"
)

# Check quality
quality = manager.quality_check(examples, ["text", "label"])
print(f"Overall quality: {quality['overall_quality']:.2%}")
```

### RAG Datasets

```python
# Load JSONL format
examples = manager.load_from_jsonl("rag_dataset.jsonl")

# Get statistics
stats = manager.get_dataset_stats(examples)
print(f"Average context length: {stats.avg_text_length.get('context', 0):.1f}")
```

## Best Practices

1. **Always validate data** before using in DSPy modules
2. **Check data quality metrics** to identify potential issues
3. **Use appropriate train/val/test splits** (typically 70/15/15)
4. **Document data sources and preprocessing steps**
5. **Monitor data drift** in production systems
6. **Handle missing values** appropriately for your use case

## Troubleshooting

### Common Issues

1. **Loading Errors**:
   - Check file format and encoding
   - Verify JSON structure for nested objects
   - Ensure CSV headers match expected columns

2. **Validation Failures**:
   - Review required fields specification
   - Check for empty strings vs null values
   - Validate data types match expectations

3. **Quality Issues**:
   - High missing value rates may indicate data collection problems
   - Low diversity suggests need for more varied examples
   - Inconsistent formats require preprocessing

### Performance Tips

1. **Use JSONL for large datasets** (streaming friendly)
2. **Sample data for exploration** before full processing
3. **Cache preprocessed datasets** to avoid recomputation
4. **Use appropriate data types** to minimize memory usage

## Interactive Features

### Real-time Updates

- All notebooks provide immediate feedback as you adjust parameters
- See preprocessing effects instantly with before/after comparisons
- Interactive sliders and controls for exploring data

### Visual Feedback

- Color-coded quality indicators
- Progress bars for long operations
- Clear success/error messages
- Rich HTML displays for complex data

### Export Capabilities

- Save processed datasets in multiple formats
- Export quality reports and analysis
- Generate configuration files for reproducibility

## Next Steps

After completing this module:

1. **Module 07**: Learn about tracing and debugging DSPy applications
2. **Module 08**: Build custom DSPy modules and components
3. **Apply dataset management** to your own DSPy projects

## Resources

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Marimo Documentation](https://docs.marimo.io/)
- [Data Quality Best Practices](https://example.com/data-quality)
- [Dataset Management Patterns](https://example.com/dataset-patterns)

---

**Ready to start?** Begin with the first notebook:

```bash
uv run marimo run dataset_loading_basics.py
```
