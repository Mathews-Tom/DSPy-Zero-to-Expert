"""
Solution 03: Quality Assessment

This solution demonstrates comprehensive data quality assessment techniques
for DSPy datasets, including metrics calculation, bias detection, and reporting.

Learning Objectives:
- Calculate comprehensive quality metrics
- Detect and analyze data bias
- Generate quality reports
- Implement quality improvement strategies
- Monitor data quality over time
"""

import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from dataset_management import DatasetManager
from data_quality_metrics import (
    DataQualityAnalyzer, QualityReporter, assess_dataset_quality,
    generate_quality_report
)
from data_preprocessing import DataPreprocessor, PreprocessingConfig
from dspy import Example


def solution_basic_quality_assessment():
    """
    Solution for basic quality assessment exercises
    """
    print("=== Solution 03: Quality Assessment ===\n")

    # Load sample datasets
    manager = DatasetManager()

    # Exercise 1: Basic quality metrics
    print("1. Basic quality metrics...")

    qa_examples = manager.load_from_json("data/sample_qa.json")

    # Assess basic quality
    quality_metrics = assess_dataset_quality(
        qa_examples,
        required_fields=['question', 'answer'],
        text_fields=['question', 'answer']
    )

    print(f"   ✅ Analyzed {quality_metrics.total_examples} examples")
    print(f"   Overall quality score: {quality_metrics.overall_quality_score:.2%}")
    print(f"   Quality grade: {quality_metrics.quality_grade}")
    print(f"   Valid examples: {quality_metrics.valid_examples}")

    # Show detailed metrics
    print(f"   Completeness: {quality_metrics.completeness_score:.2%}")
    print(f"   Consistency: {quality_metrics.consistency_score:.2%}")
    print(f"   Uniqueness: {quality_metrics.uniqueness_score:.2%}")
    print(f"   Text quality: {quality_metrics.text_quality_score:.2%}")
    print(f"   Bias score: {quality_metrics.bias_score:.2%}")

    # Exercise 2: Field-specific analysis
    print("\n2. Field-specific analysis...")

    # Analyze each field separately
    for field, completeness in quality_metrics.field_completeness.items():
        avg_length = quality_metrics.avg_text_length.get(field, 0)
        diversity = quality_metrics.field_diversity.get(field, 0)

        print(f"   {field}:")
        print(f"     Completeness: {completeness:.2%}")
        print(f"     Avg length: {avg_length:.1f} characters")
        print(f"     Diversity: {diversity:.2%}")

    # Exercise 3: Quality issues identification
    print("\n3. Quality issues identification...")

    if quality_metrics.issues:
        print("   Issues found:")
        for i, issue in enumerate(quality_metrics.issues, 1):
            print(f"     {i}. {issue}")
    else:
        print("   ✅ No quality issues found!")

    if quality_metrics.recommendations:
        print("   Recommendations:")
        for i, rec in enumerate(quality_metrics.recommendations, 1):
            print(f"     {i}. {rec}")

    # Exercise 4: Compare different datasets
    print("\n4. Compare different datasets...")

    # Load classification dataset
    classification_examples = manager.load_from_csv(
        "data/sample_classification.csv", "text", "label"
    )

    classification_quality = assess_dataset_quality(
        classification_examples,
        required_fields=['text', 'label'],
        text_fields=['text']
    )

    # Load RAG dataset
    rag_examples = manager.load_from_jsonl("data/sample_rag.jsonl")

    rag_quality = assess_dataset_quality(
        rag_examples,
        required_fields=['question', 'context', 'answer'],
        text_fields=['question', 'context', 'answer']
    )

    # Compare quality scores
    datasets = [
        ("QA Dataset", quality_metrics),
        ("Classification Dataset", classification_quality),
        ("RAG Dataset", rag_quality)
    ]

    print("   Quality comparison:")
    for name, metrics in datasets:
        print(f"     {name}: {metrics.overall_quality_score:.2%} (Grade {metrics.quality_grade})")

    print("\n=== Basic Quality Assessment Complete ===")
    return quality_metrics


def solution_advanced_quality_analysis():
    """
    Solution for advanced quality analysis techniques
    """
    print("\n=== Advanced Quality Analysis ===\n")

    manager = DatasetManager()
    analyzer = DataQualityAnalyzer()

    # Exercise 1: Detailed completeness analysis
    print("1. Detailed completeness analysis...")

    qa_examples = manager.load_from_json("data/sample_qa.json")

    completeness_analysis = analyzer.analyze_completeness(
        qa_examples, ['question', 'answer', 'category', 'difficulty']
    )

    print("   Field completeness:")
    for field, score in completeness_analysis['field_completeness'].items():
        print(f"     {field}: {score:.2%}")

    print(f"   Overall missing rate: {completeness_analysis['missing_rate']:.2%}")

    # Exercise 2: Consistency analysis
    print("\n2. Consistency analysis...")

    consistency_analysis = analyzer.analyze_consistency(
        qa_examples, ['question', 'answer']
    )

    print("   Type consistency:")
    for field, score in consistency_analysis['type_consistency'].items():
        print(f"     {field}: {score:.2%}")

    print("   Format consistency:")
    for field, score in consistency_analysis['format_consistency'].items():
        print(f"     {field}: {score:.2%}")

    # Exercise 3: Uniqueness and diversity analysis
    print("\n3. Uniqueness and diversity analysis...")

    uniqueness_analysis = analyzer.analyze_uniqueness(
        qa_examples, ['question', 'answer']
    )

    print(f"   Duplicate rate: {uniqueness_analysis['duplicate_rate']:.2%}")
    print("   Field diversity:")
    for field, score in uniqueness_analysis['field_diversity'].items():
        print(f"     {field}: {score:.2%}")

    # Exercise 4: Text quality analysis
    print("\n4. Text quality analysis...")

    text_quality_analysis = analyzer.analyze_text_quality(
        qa_examples, ['question', 'answer']
    )

    print("   Average text lengths:")
    for field, length in text_quality_analysis['avg_text_length'].items():
        print(f"     {field}: {length:.1f} characters")

    print("   Readability scores:")
    for field, score in text_quality_analysis['readability_scores'].items():
        print(f"     {field}: {score:.2%}")

    # Exercise 5: Bias detection
    print("\n5. Bias detection...")

    bias_analysis = analyzer.analyze_bias(qa_examples, ['question', 'answer'])

    print("   Length bias (higher is better):")
    for field, score in bias_analysis['length_bias'].items():
        print(f"     {field}: {score:.2%}")

    print("   Vocabulary bias (higher is better):")
    for field, score in bias_analysis['vocabulary_bias'].items():
        print(f"     {field}: {score:.2%}")

    print(f"   Overall bias score: {bias_analysis['bias_score']:.2%}")

    print("\n=== Advanced Quality Analysis Complete ===")


def solution_quality_improvement():
    """
    Solution for quality improvement strategies
    """
    print("\n=== Quality Improvement Strategies ===\n")

    manager = DatasetManager()

    # Exercise 1: Before and after quality comparison
    print("1. Before and after quality comparison...")

    # Load original data
    original_examples = manager.load_from_json("data/sample_qa.json")

    # Assess original quality
    original_quality = assess_dataset_quality(
        original_examples,
        required_fields=['question', 'answer'],
        text_fields=['question', 'answer']
    )

    print(f"   Original quality: {original_quality.overall_quality_score:.2%}")

    # Apply preprocessing to improve quality
    config = PreprocessingConfig(
        remove_extra_whitespace=True,
        normalize_unicode=True,
        normalize_quotes=True,
        normalize_dashes=True,
        min_text_length=15,
        min_word_count=3,
        remove_duplicates=True
    )

    preprocessor = DataPreprocessor(config)
    improved_examples = preprocessor.preprocess_dataset(
        original_examples, ['question', 'answer']
    )

    # Assess improved quality
    improved_quality = assess_dataset_quality(
        improved_examples,
        required_fields=['question', 'answer'],
        text_fields=['question', 'answer']
    )

    print(f"   Improved quality: {improved_quality.overall_quality_score:.2%}")

    improvement = improved_quality.overall_quality_score - original_quality.overall_quality_score
    print(f"   Quality improvement: {improvement:+.2%}")

    # Exercise 2: Targeted quality improvements
    print("\n2. Targeted quality improvements...")

    # Create examples with specific quality issues
    problematic_examples = [
        Example(question="What is Python?", answer="A programming language."),
        Example(question="What is Python?", answer="A programming language."),  # Duplicate
        Example(question="?", answer="Short."),  # Too short
        Example(question="What is machine learning?", answer=""),  # Empty answer
        Example(question="  What   is    AI?  ", answer="  Artificial intelligence.  "),  # Whitespace
        Example(question="What's the "best" approach?", answer="It depends—many factors."),  # Quotes/dashes
    ]

    print(f"   Original problematic examples: {len(problematic_examples)}")

    # Apply targeted improvements
    targeted_config = PreprocessingConfig(
        remove_extra_whitespace=True,
        normalize_quotes=True,
        normalize_dashes=True,
        min_text_length=10,
        min_word_count=2,
        remove_duplicates=True
    )

    targeted_preprocessor = DataPreprocessor(targeted_config)
    fixed_examples = targeted_preprocessor.preprocess_dataset(
        problematic_examples, ['question', 'answer']
    )

    print(f"   Fixed examples: {len(fixed_examples)}")
    print(f"   Issues resolved: {len(problematic_examples) - len(fixed_examples)}")

    # Show specific fixes
    if fixed_examples:
        print("   Example fixes:")
        original_whitespace = "  What   is    AI?  "
        fixed_whitespace = fixed_examples[0].question if len(fixed_examples) > 0 else "N/A"
        print(f"     Whitespace: '{original_whitespace}' -> '{fixed_whitespace}'")

    # Exercise 3: Quality monitoring over time
    print("\n3. Quality monitoring over time...")

    # Simulate quality monitoring with different dataset versions
    dataset_versions = [
        ("v1.0", original_examples[:5]),
        ("v1.1", original_examples[:8]),
        ("v1.2", original_examples),
        ("v2.0", improved_examples)
    ]

    print("   Quality trends:")
    for version, examples in dataset_versions:
        quality = assess_dataset_quality(
            examples,
            required_fields=['question', 'answer'],
            text_fields=['question', 'answer']
        )
        print(f"     {version}: {quality.overall_quality_score:.2%} "
              f"({len(examples)} examples, Grade {quality.quality_grade})")

    # Exercise 4: Quality thresholds and alerts
    print("\n4. Quality thresholds and alerts...")

    # Define quality thresholds
    thresholds = {
        'excellent': 0.90,
        'good': 0.75,
        'acceptable': 0.60,
        'poor': 0.40
    }

    # Check quality against thresholds
    current_quality = improved_quality.overall_quality_score

    print(f"   Current quality: {current_quality:.2%}")

    for level, threshold in thresholds.items():
        if current_quality >= threshold:
            print(f"   ✅ Quality level: {level.upper()}")
            break
    else:
        print("   ❌ Quality level: UNACCEPTABLE")

    # Generate alerts for specific issues
    alerts = []
    if improved_quality.completeness_score < 0.8:
        alerts.append("LOW COMPLETENESS: Consider data collection improvements")
    if improved_quality.duplicate_rate > 0.1:
        alerts.append("HIGH DUPLICATES: Implement deduplication process")
    if improved_quality.bias_score < 0.5:
        alerts.append("POTENTIAL BIAS: Review data sources for balance")

    if alerts:
        print("   🚨 Quality alerts:")
        for alert in alerts:
            print(f"     - {alert}")
    else:
        print("   ✅ No quality alerts")

    print("\n=== Quality Improvement Complete ===")


def solution_quality_reporting():
    """
    Solution for quality reporting exercises
    """
    print("\n=== Quality Reporting ===\n")

    manager = DatasetManager()

    # Exercise 1: Generate comprehensive quality report
    print("1. Generate comprehensive quality report...")

    qa_examples = manager.load_from_json("data/sample_qa.json")
    quality_metrics = assess_dataset_quality(
        qa_examples,
        required_fields=['question', 'answer'],
        text_fields=['question', 'answer']
    )

    # Generate HTML report
    html_report = generate_quality_report(
        quality_metrics,
        "Sample QA Dataset",
        "html"
    )

    # Save HTML report
    with open("quality_report.html", "w", encoding='utf-8') as f:
        f.write(html_report)

    print("   ✅ Generated HTML quality report: quality_report.html")

    # Generate text report
    text_report = generate_quality_report(
        quality_metrics,
        "Sample QA Dataset",
        "text"
    )

    # Save text report
    with open("quality_report.txt", "w", encoding='utf-8') as f:
        f.write(text_report)

    print("   ✅ Generated text quality report: quality_report.txt")

    # Exercise 2: Custom quality dashboard
    print("\n2. Custom quality dashboard...")

    # Create a simple dashboard summary
    dashboard_data = {
        'dataset_name': 'Sample QA Dataset',
        'total_examples': quality_metrics.total_examples,
        'quality_score': quality_metrics.overall_quality_score,
        'quality_grade': quality_metrics.quality_grade,
        'issues_count': len(quality_metrics.issues),
        'recommendations_count': len(quality_metrics.recommendations)
    }

    print("   📊 Quality Dashboard:")
    print(f"     Dataset: {dashboard_data['dataset_name']}")
    print(f"     Examples: {dashboard_data['total_examples']:,}")
    print(f"     Quality: {dashboard_data['quality_score']:.1%} (Grade {dashboard_data['quality_grade']})")
    print(f"     Issues: {dashboard_data['issues_count']}")
    print(f"     Recommendations: {dashboard_data['recommendations_count']}")

    # Exercise 3: Quality comparison report
    print("\n3. Quality comparison report...")

    # Compare multiple datasets
    datasets = [
        ("QA Dataset", manager.load_from_json("data/sample_qa.json")),
        ("Classification", manager.load_from_csv("data/sample_classification.csv", "text", "label")),
        ("RAG Dataset", manager.load_from_jsonl("data/sample_rag.jsonl"))
    ]

    comparison_results = []

    for name, examples in datasets:
        if name == "Classification":
            quality = assess_dataset_quality(examples, ['text', 'label'], ['text'])
        elif name == "RAG Dataset":
            quality = assess_dataset_quality(examples, ['question', 'context', 'answer'], ['question', 'context', 'answer'])
        else:
            quality = assess_dataset_quality(examples, ['question', 'answer'], ['question', 'answer'])

        comparison_results.append((name, quality))

    print("   📊 Dataset Quality Comparison:")
    print("   " + "-" * 60)
    print("   Dataset               | Examples | Quality | Grade | Issues")
    print("   " + "-" * 60)

    for name, quality in comparison_results:
        print(f"   {name:<20} | {quality.total_examples:>8} | {quality.overall_quality_score:>6.1%} | {quality.quality_grade:>5} | {len(quality.issues):>6}")

    print("   " + "-" * 60)

    print("\n=== Quality Reporting Complete ===")


if __name__ == "__main__":
    # Run basic quality assessment solution
    basic_quality = solution_basic_quality_assessment()

    # Run advanced quality analysis solution
    solution_advanced_quality_analysis()

    # Run quality improvement solution
    solution_quality_improvement()

    # Run quality reporting solution
    solution_quality_reporting()

    print("\n🎉 All quality assessment exercises completed successfully!")
    print("\nKey takeaways:")
    print("- Quality assessment should cover multiple dimensions")
    print("- Use specific metrics for completeness, consistency, and uniqueness")
    print("- Detect and address bias in your datasets")
    print("- Monitor quality improvements after preprocessing")
    print("- Generate comprehensive reports for stakeholders")
    print("- Set quality thresholds and alerts for monitoring")
    print("- Compare quality across different datasets and versions")