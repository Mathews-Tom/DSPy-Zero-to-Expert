"""
Solution 04: Data Exploration

This solution demonstrates comprehensive data exploration techniques
for DSPy datasets using interactive tools and statistical analysis.

Learning Objectives:
- Perform statistical analysis of datasets
- Create data visualizations
- Explore data patterns and distributions
- Identify data insights and anomalies
- Build interactive exploration tools
"""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from data_quality_metrics import DataQualityAnalyzer
from dataset_management import DatasetManager
from dspy import Example


def solution_basic_data_exploration():
    """
    Solution for basic data exploration exercises
    """
    print("=== Solution 04: Data Exploration ===\n")

    manager = DatasetManager()

    # Exercise 1: Dataset overview and statistics
    print("1. Dataset overview and statistics...")

    # Load QA dataset
    qa_examples = manager.load_from_json("data/sample_qa.json")

    print(f"   ✅ Loaded {len(qa_examples)} QA examples")

    # Basic statistics
    stats = manager.get_dataset_stats(qa_examples)

    print(f"   Total examples: {stats.total_examples}")
    print(f"   Fields found: {list(stats.field_counts.keys())}")

    # Field completeness
    print("   Field completeness:")
    for field, count in stats.field_counts.items():
        completeness = count / stats.total_examples * 100
        print(f"     {field}: {count}/{stats.total_examples} ({completeness:.1f}%)")

    # Text length analysis
    print("   Average text lengths:")
    for field, avg_length in stats.avg_text_length.items():
        if avg_length > 0:
            print(f"     {field}: {avg_length:.1f} characters")

    # Exercise 2: Field distribution analysis
    print("\n2. Field distribution analysis...")

    # Analyze categorical fields
    categorical_fields = ["category", "difficulty"]

    for field in categorical_fields:
        if field in stats.field_counts:
            values = []
            for example in qa_examples:
                if hasattr(example, field):
                    value = getattr(example, field)
                    if value is not None:
                        values.append(value)

            if values:
                distribution = Counter(values)
                print(f"   {field} distribution:")
                for value, count in distribution.most_common():
                    percentage = count / len(values) * 100
                    print(f"     {value}: {count} ({percentage:.1f}%)")

    # Exercise 3: Text length distribution
    print("\n3. Text length distribution...")

    text_fields = ["question", "answer"]

    for field in text_fields:
        lengths = []
        for example in qa_examples:
            if hasattr(example, field):
                value = getattr(example, field)
                if value and isinstance(value, str):
                    lengths.append(len(value))

        if lengths:
            print(f"   {field} length statistics:")
            print(f"     Min: {min(lengths)} characters")
            print(f"     Max: {max(lengths)} characters")
            print(f"     Mean: {statistics.mean(lengths):.1f} characters")
            print(f"     Median: {statistics.median(lengths):.1f} characters")
            if len(lengths) > 1:
                print(f"     Std Dev: {statistics.stdev(lengths):.1f} characters")

    # Exercise 4: Word count analysis
    print("\n4. Word count analysis...")

    for field in text_fields:
        word_counts = []
        for example in qa_examples:
            if hasattr(example, field):
                value = getattr(example, field)
                if value and isinstance(value, str):
                    word_count = len(value.split())
                    word_counts.append(word_count)

        if word_counts:
            print(f"   {field} word count statistics:")
            print(f"     Min: {min(word_counts)} words")
            print(f"     Max: {max(word_counts)} words")
            print(f"     Mean: {statistics.mean(word_counts):.1f} words")
            print(f"     Median: {statistics.median(word_counts):.1f} words")

    print("\n=== Basic Data Exploration Complete ===")
    return qa_examples


def solution_advanced_exploration():
    """
    Solution for advanced data exploration techniques
    """
    print("\n=== Advanced Data Exploration ===\n")

    manager = DatasetManager()

    # Exercise 1: Multi-dataset comparison
    print("1. Multi-dataset comparison...")

    # Load different datasets
    datasets = {
        "QA": manager.load_from_json("data/sample_qa.json"),
        "Classification": manager.load_from_csv(
            "data/sample_classification.csv", "text", "label"
        ),
        "RAG": manager.load_from_jsonl("data/sample_rag.jsonl"),
    }

    print("   Dataset comparison:")
    print("   " + "-" * 50)
    print("   Dataset        | Examples | Avg Text Length")
    print("   " + "-" * 50)

    for name, examples in datasets.items():
        if not examples:
            continue

        # Calculate average text length across all text fields
        total_length = 0
        text_count = 0

        for example in examples:
            for key, value in example.__dict__.items():
                if not key.startswith("_") and isinstance(value, str):
                    total_length += len(value)
                    text_count += 1

        avg_length = total_length / text_count if text_count > 0 else 0
        print(f"   {name:<14} | {len(examples):>8} | {avg_length:>13.1f}")

    print("   " + "-" * 50)

    # Exercise 2: Pattern detection
    print("\n2. Pattern detection...")

    qa_examples = datasets["QA"]

    # Detect question patterns
    question_starters = defaultdict(int)
    question_enders = defaultdict(int)

    for example in qa_examples:
        if hasattr(example, "question") and example.question:
            question = example.question.strip()

            # First word
            first_word = question.split()[0].lower() if question.split() else ""
            question_starters[first_word] += 1

            # Last character
            last_char = question[-1] if question else ""
            question_enders[last_char] += 1

    print("   Question starter patterns:")
    for starter, count in question_starters.most_common(5):
        print(f"     '{starter}': {count} questions")

    print("   Question ending patterns:")
    for ender, count in question_enders.most_common():
        print(f"     '{ender}': {count} questions")

    # Exercise 3: Vocabulary analysis
    print("\n3. Vocabulary analysis...")

    # Analyze vocabulary in questions and answers
    vocabulary = defaultdict(int)
    field_vocabularies = defaultdict(lambda: defaultdict(int))

    import re

    for example in qa_examples:
        for field in ["question", "answer"]:
            if hasattr(example, field):
                text = getattr(example, field)
                if text and isinstance(text, str):
                    # Extract words (simple tokenization)
                    words = re.findall(r"\b\w+\b", text.lower())
                    for word in words:
                        vocabulary[word] += 1
                        field_vocabularies[field][word] += 1

    print(f"   Total unique words: {len(vocabulary)}")
    print(f"   Total word occurrences: {sum(vocabulary.values())}")

    print("   Most common words overall:")
    for word, count in vocabulary.most_common(10):
        print(f"     {word}: {count}")

    # Compare vocabularies between fields
    question_words = set(field_vocabularies["question"].keys())
    answer_words = set(field_vocabularies["answer"].keys())

    shared_words = question_words & answer_words
    question_only = question_words - answer_words
    answer_only = answer_words - question_words

    print(f"   Vocabulary overlap:")
    print(f"     Shared words: {len(shared_words)}")
    print(f"     Question-only words: {len(question_only)}")
    print(f"     Answer-only words: {len(answer_only)}")

    # Exercise 4: Anomaly detection
    print("\n4. Anomaly detection...")

    # Detect unusually long or short texts
    text_lengths = []
    for example in qa_examples:
        for field in ["question", "answer"]:
            if hasattr(example, field):
                text = getattr(example, field)
                if text and isinstance(text, str):
                    text_lengths.append((field, len(text), text[:50] + "..."))

    if text_lengths:
        # Sort by length
        text_lengths.sort(key=lambda x: x[1])

        print("   Shortest texts:")
        for field, length, preview in text_lengths[:3]:
            print(f"     {field} ({length} chars): {preview}")

        print("   Longest texts:")
        for field, length, preview in text_lengths[-3:]:
            print(f"     {field} ({length} chars): {preview}")

        # Detect outliers using IQR method
        lengths_only = [x[1] for x in text_lengths]
        q1 = statistics.quantiles(lengths_only, n=4)[0]
        q3 = statistics.quantiles(lengths_only, n=4)[2]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = [x for x in text_lengths if x[1] < lower_bound or x[1] > upper_bound]
        print(f"   Statistical outliers: {len(outliers)} texts")

    print("\n=== Advanced Data Exploration Complete ===")


def solution_interactive_exploration():
    """
    Solution for interactive exploration techniques
    """
    print("\n=== Interactive Data Exploration ===\n")

    manager = DatasetManager()

    # Exercise 1: Interactive data filtering
    print("1. Interactive data filtering...")

    qa_examples = manager.load_from_json("data/sample_qa.json")

    # Simulate interactive filtering
    filters = {
        "min_question_length": 20,
        "max_question_length": 100,
        "categories": ["programming", "ai"],
        "difficulties": ["beginner", "intermediate"],
    }

    filtered_examples = []
    for example in qa_examples:
        # Apply filters
        if hasattr(example, "question") and example.question:
            question_length = len(example.question)
            if (
                question_length < filters["min_question_length"]
                or question_length > filters["max_question_length"]
            ):
                continue

        if hasattr(example, "category") and example.category:
            if example.category not in filters["categories"]:
                continue

        if hasattr(example, "difficulty") and example.difficulty:
            if example.difficulty not in filters["difficulties"]:
                continue

        filtered_examples.append(example)

    print(f"   Original examples: {len(qa_examples)}")
    print(f"   Filtered examples: {len(filtered_examples)}")
    print(f"   Filter criteria:")
    for key, value in filters.items():
        print(f"     {key}: {value}")

    # Exercise 2: Data sampling and preview
    print("\n2. Data sampling and preview...")

    # Sample different portions of the data
    sample_sizes = [3, 5, 10]

    for size in sample_sizes:
        if size <= len(qa_examples):
            sampled = manager.sample_examples(qa_examples, size, random_seed=42)
            print(f"   Sample of {size} examples:")

            for i, example in enumerate(sampled):
                question = getattr(example, "question", "N/A")
                category = getattr(example, "category", "N/A")
                print(f"     {i+1}. [{category}] {question[:40]}...")

    # Exercise 3: Field correlation analysis
    print("\n3. Field correlation analysis...")

    # Analyze relationships between fields
    category_difficulty = defaultdict(lambda: defaultdict(int))
    category_length = defaultdict(list)

    for example in qa_examples:
        category = getattr(example, "category", "unknown")
        difficulty = getattr(example, "difficulty", "unknown")

        category_difficulty[category][difficulty] += 1

        if hasattr(example, "question") and example.question:
            category_length[category].append(len(example.question))

    print("   Category-Difficulty correlation:")
    for category, difficulties in category_difficulty.items():
        print(f"     {category}:")
        for difficulty, count in difficulties.items():
            print(f"       {difficulty}: {count}")

    print("   Category-Length correlation:")
    for category, lengths in category_length.items():
        if lengths:
            avg_length = statistics.mean(lengths)
            print(f"     {category}: {avg_length:.1f} avg chars")

    # Exercise 4: Data export for visualization
    print("\n4. Data export for visualization...")

    # Prepare data for external visualization tools
    export_data = []

    for i, example in enumerate(qa_examples):
        record = {
            "id": i,
            "question_length": len(getattr(example, "question", "")),
            "answer_length": len(getattr(example, "answer", "")),
            "category": getattr(example, "category", "unknown"),
            "difficulty": getattr(example, "difficulty", "unknown"),
            "question_words": len(getattr(example, "question", "").split()),
            "answer_words": len(getattr(example, "answer", "").split()),
        }
        export_data.append(record)

    # Save as JSON for visualization
    with open("exploration_data.json", "w") as f:
        json.dump(export_data, f, indent=2)

    print("   ✅ Exported data to exploration_data.json")

    # Create summary statistics for visualization
    summary_stats = {
        "total_examples": len(qa_examples),
        "categories": list(
            set(getattr(ex, "category", "unknown") for ex in qa_examples)
        ),
        "difficulties": list(
            set(getattr(ex, "difficulty", "unknown") for ex in qa_examples)
        ),
        "avg_question_length": statistics.mean(
            [len(getattr(ex, "question", "")) for ex in qa_examples]
        ),
        "avg_answer_length": statistics.mean(
            [len(getattr(ex, "answer", "")) for ex in qa_examples]
        ),
    }

    with open("summary_stats.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    print("   ✅ Exported summary statistics to summary_stats.json")

    print("\n=== Interactive Data Exploration Complete ===")


def solution_exploration_insights():
    """
    Solution for generating insights from data exploration
    """
    print("\n=== Data Exploration Insights ===\n")

    manager = DatasetManager()

    # Exercise 1: Generate dataset insights
    print("1. Generate dataset insights...")

    qa_examples = manager.load_from_json("data/sample_qa.json")

    insights = []

    # Insight 1: Dataset size and coverage
    total_examples = len(qa_examples)
    categories = set(getattr(ex, "category", "unknown") for ex in qa_examples)
    difficulties = set(getattr(ex, "difficulty", "unknown") for ex in qa_examples)

    insights.append(
        f"Dataset contains {total_examples} examples across {len(categories)} categories"
    )
    insights.append(f"Difficulty levels covered: {', '.join(sorted(difficulties))}")

    # Insight 2: Text characteristics
    question_lengths = [len(getattr(ex, "question", "")) for ex in qa_examples]
    answer_lengths = [len(getattr(ex, "answer", "")) for ex in qa_examples]

    avg_q_len = statistics.mean(question_lengths)
    avg_a_len = statistics.mean(answer_lengths)

    insights.append(f"Average question length: {avg_q_len:.1f} characters")
    insights.append(f"Average answer length: {avg_a_len:.1f} characters")

    if avg_a_len > avg_q_len * 2:
        insights.append(
            "Answers are significantly longer than questions (good for detailed responses)"
        )

    # Insight 3: Category distribution
    category_counts = Counter(getattr(ex, "category", "unknown") for ex in qa_examples)
    most_common_category = category_counts.most_common(1)[0]

    insights.append(
        f"Most common category: {most_common_category[0]} ({most_common_category[1]} examples)"
    )

    # Check for balance
    category_values = list(category_counts.values())
    if max(category_values) > min(category_values) * 3:
        insights.append("Dataset shows category imbalance - consider balancing")
    else:
        insights.append("Dataset categories are reasonably balanced")

    # Insight 4: Quality indicators
    empty_answers = sum(
        1 for ex in qa_examples if not getattr(ex, "answer", "").strip()
    )
    short_questions = sum(
        1 for ex in qa_examples if len(getattr(ex, "question", "")) < 10
    )

    if empty_answers > 0:
        insights.append(f"Found {empty_answers} examples with empty answers")

    if short_questions > 0:
        insights.append(f"Found {short_questions} examples with very short questions")

    print("   📊 Dataset Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"     {i}. {insight}")

    # Exercise 2: Recommendations based on exploration
    print("\n2. Recommendations based on exploration...")

    recommendations = []

    # Size recommendations
    if total_examples < 50:
        recommendations.append(
            "Consider expanding dataset size for better model training"
        )
    elif total_examples > 1000:
        recommendations.append(
            "Large dataset - consider sampling for faster experimentation"
        )

    # Balance recommendations
    if len(categories) < 3:
        recommendations.append("Add more categories to increase dataset diversity")

    # Quality recommendations
    if empty_answers > total_examples * 0.1:
        recommendations.append(
            "High rate of empty answers - review data collection process"
        )

    if short_questions > total_examples * 0.2:
        recommendations.append(
            "Many short questions - consider minimum length requirements"
        )

    # Text length recommendations
    if avg_q_len < 20:
        recommendations.append(
            "Questions are quite short - consider more detailed questions"
        )

    if avg_a_len < 30:
        recommendations.append(
            "Answers are quite short - consider more comprehensive answers"
        )

    print("   💡 Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"     {i}. {rec}")

    # Exercise 3: Export exploration report
    print("\n3. Export exploration report...")

    # Create comprehensive exploration report
    report = {
        "dataset_name": "Sample QA Dataset",
        "exploration_date": "2024-01-01",
        "basic_stats": {
            "total_examples": total_examples,
            "categories": list(categories),
            "difficulties": list(difficulties),
            "avg_question_length": avg_q_len,
            "avg_answer_length": avg_a_len,
        },
        "insights": insights,
        "recommendations": recommendations,
        "quality_flags": {
            "empty_answers": empty_answers,
            "short_questions": short_questions,
        },
    }

    # Save exploration report
    with open("exploration_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("   ✅ Saved exploration report to exploration_report.json")

    # Create human-readable report
    readable_report = f"""
DATA EXPLORATION REPORT
======================

Dataset: {report['dataset_name']}
Date: {report['exploration_date']}

BASIC STATISTICS
---------------
Total Examples: {report['basic_stats']['total_examples']}
Categories: {', '.join(report['basic_stats']['categories'])}
Difficulties: {', '.join(report['basic_stats']['difficulties'])}
Avg Question Length: {report['basic_stats']['avg_question_length']:.1f} chars
Avg Answer Length: {report['basic_stats']['avg_answer_length']:.1f} chars

INSIGHTS
--------
"""

    for i, insight in enumerate(report["insights"], 1):
        readable_report += f"{i}. {insight}\n"

    readable_report += "\nRECOMMENDATIONS\n--------------\n"

    for i, rec in enumerate(report["recommendations"], 1):
        readable_report += f"{i}. {rec}\n"

    # Save readable report
    with open("exploration_report.txt", "w") as f:
        f.write(readable_report)

    print("   ✅ Saved readable report to exploration_report.txt")

    print("\n=== Data Exploration Insights Complete ===")


if __name__ == "__main__":
    # Run basic data exploration solution
    qa_data = solution_basic_data_exploration()

    # Run advanced exploration solution
    solution_advanced_exploration()

    # Run interactive exploration solution
    solution_interactive_exploration()

    # Run exploration insights solution
    solution_exploration_insights()

    print("\n🎉 All data exploration exercises completed successfully!")
    print("\nKey takeaways:")
    print("- Start with basic statistics to understand your data")
    print("- Look for patterns in categorical and numerical fields")
    print("- Analyze text characteristics and vocabulary")
    print("- Detect anomalies and outliers in your data")
    print("- Use interactive filtering to explore subsets")
    print("- Generate actionable insights and recommendations")
    print("- Export data and reports for further analysis")
    print("- Document your exploration process and findings")
