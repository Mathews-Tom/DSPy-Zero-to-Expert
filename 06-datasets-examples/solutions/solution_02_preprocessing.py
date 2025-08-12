"""
Solution 02: Data Preprocessing

This solution demonstrates comprehensive data preprocessing techniques
for DSPy datasets, including cleaning, normalization, and augmentation.

Learning Objectives:
- Apply text cleaning and normalization
- Configure preprocessing pipelines
- Handle data quality issues
- Implement data augmentation
- Validate preprocessing results
"""

import sys
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from data_preprocessing import (
    DataAugmentor,
    DataPreprocessor,
    DataValidator,
    PreprocessingConfig,
    preprocess_classification_dataset,
    preprocess_qa_dataset,
    preprocess_rag_dataset,
)
from dataset_management import DatasetManager
from dspy import Example  # pylint: disable=import-error


def solution_basic_preprocessing():
    """
    Solution for basic preprocessing exercises
    """
    print("=== Solution 02: Data Preprocessing ===\n")

    # Load sample data
    manager = DatasetManager()

    # Exercise 1: Basic text cleaning
    print("1. Basic text cleaning...")

    # Create examples with messy text
    messy_examples = [
        Example(
            question="  What   is    Python???  ",
            answer="Python is a   programming language.  \n\n  ",
        ),
        Example(
            question="<p>How to use <b>DSPy</b>?</p>",
            answer="DSPy is a framework for   LLM programming.",
        ),
        Example(
            question='What\'s the "best" way to learn?',
            answer="Practice and — persistence are key!",
        ),
    ]

    # Configure basic cleaning
    config = PreprocessingConfig(
        remove_extra_whitespace=True,
        normalize_unicode=True,
        remove_html_tags=True,
        normalize_quotes=True,
        normalize_dashes=True,
    )

    preprocessor = DataPreprocessor(config)
    cleaned_examples = preprocessor.preprocess_dataset(
        messy_examples, ["question", "answer"]
    )

    print(f"   ✅ Cleaned {len(cleaned_examples)} examples")

    # Show before/after comparison
    if messy_examples and cleaned_examples:
        print("   Before:", repr(messy_examples[0].question))
        print("   After: ", repr(cleaned_examples[0].question))

    # Exercise 2: Advanced preprocessing configuration
    print("\n2. Advanced preprocessing configuration...")

    # Load real dataset
    qa_examples = manager.load_from_json("data/sample_qa.json")

    # Configure advanced preprocessing
    advanced_config = PreprocessingConfig(
        remove_extra_whitespace=True,
        normalize_unicode=True,
        normalize_quotes=True,
        normalize_dashes=True,
        min_text_length=10,
        min_word_count=3,
        max_text_length=1000,
        remove_duplicates=True,
    )

    # Apply preprocessing
    preprocessed_qa = preprocess_qa_dataset(qa_examples, advanced_config)

    print(f"   Original examples: {len(qa_examples)}")
    print(f"   Preprocessed examples: {len(preprocessed_qa)}")
    print(f"   Filtered out: {len(qa_examples) - len(preprocessed_qa)}")

    # Exercise 3: Field-specific preprocessing
    print("\n3. Field-specific preprocessing...")

    # Create examples with different field requirements
    mixed_examples = [
        Example(
            title="INTRODUCTION TO PYTHON",
            content="python is a programming language",
            tags="programming, python, beginner",
        ),
        Example(
            title="advanced machine learning",
            content="MACHINE LEARNING IS COMPLEX",
            tags="ML, AI, Advanced",
        ),
    ]

    # Different configs for different fields
    title_config = PreprocessingConfig(lowercase=True, remove_extra_whitespace=True)

    content_config = PreprocessingConfig(
        normalize_unicode=True, remove_extra_whitespace=True
    )

    # Process title field
    title_preprocessor = DataPreprocessor(title_config)
    for example in mixed_examples:
        if hasattr(example, "title"):
            original_title = example.title
            cleaned_title = title_preprocessor.clean_text(original_title)
            print(f"   Title: '{original_title}' -> '{cleaned_title}'")

    # Exercise 4: Quality filtering
    print("\n4. Quality filtering...")

    # Create examples with quality issues
    quality_examples = [
        Example(
            question="Good question here?",
            answer="Detailed answer with proper explanation.",
        ),
        Example(question="?", answer="No."),  # Too short
        Example(question="What is this" * 50, answer="Very long question"),  # Too long
        Example(question="Valid question?", answer=""),  # Empty answer
        Example(question="Another good question?", answer="Another good answer here."),
    ]

    # Configure quality filters
    quality_config = PreprocessingConfig(
        min_text_length=5, max_text_length=200, min_word_count=2, remove_duplicates=True
    )

    quality_preprocessor = DataPreprocessor(quality_config)
    filtered_examples = quality_preprocessor.preprocess_dataset(
        quality_examples, ["question", "answer"]
    )

    print(f"   Original examples: {len(quality_examples)}")
    print(f"   Quality-filtered examples: {len(filtered_examples)}")

    # Show which examples passed
    for i, example in enumerate(quality_examples):
        passed = any(
            ex.question == example.question and ex.answer == example.answer
            for ex in filtered_examples
        )
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   Example {i+1}: {status}")

    # Exercise 5: Duplicate removal
    print("\n5. Duplicate removal...")

    # Create examples with duplicates
    duplicate_examples = [
        Example(question="What is Python?", answer="A programming language"),
        Example(
            question="What is Python?", answer="A programming language"
        ),  # Exact duplicate
        Example(
            question="what is python?", answer="a programming language"
        ),  # Case difference
        Example(question="What is Java?", answer="Another programming language"),
        Example(
            question="What is Python?", answer="A programming language"
        ),  # Another duplicate
    ]

    # Configure duplicate removal
    dedup_config = PreprocessingConfig(
        remove_duplicates=True, lowercase=False  # Keep original case for comparison
    )

    dedup_preprocessor = DataPreprocessor(dedup_config)
    unique_examples = dedup_preprocessor.preprocess_dataset(
        duplicate_examples, ["question", "answer"]
    )

    print(f"   Original examples: {len(duplicate_examples)}")
    print(f"   Unique examples: {len(unique_examples)}")
    print(f"   Duplicates removed: {len(duplicate_examples) - len(unique_examples)}")

    print("\n=== Basic Preprocessing Complete ===")
    return preprocessed_qa


def solution_advanced_preprocessing():
    """
    Solution for advanced preprocessing techniques
    """
    print("\n=== Advanced Preprocessing Techniques ===\n")

    manager = DatasetManager()

    # Exercise 1: Custom text cleaning
    print("1. Custom text cleaning...")

    # Load classification data
    classification_examples = manager.load_from_csv(
        "data/sample_classification.csv", "text", "label"
    )

    # Configure aggressive cleaning for sentiment analysis
    sentiment_config = PreprocessingConfig(
        remove_extra_whitespace=True,
        normalize_unicode=True,
        remove_html_tags=True,
        normalize_quotes=True,
        lowercase=False,  # Keep case for sentiment
        remove_punctuation=False,  # Keep punctuation for sentiment
        min_text_length=5,
        remove_duplicates=True,
    )

    cleaned_sentiment = preprocess_classification_dataset(
        classification_examples, sentiment_config
    )

    print(f"   ✅ Processed {len(cleaned_sentiment)} sentiment examples")

    # Show cleaning results
    if classification_examples and cleaned_sentiment:
        original = classification_examples[0].text
        cleaned = cleaned_sentiment[0].text
        print(f"   Original: {original[:50]}...")
        print(f"   Cleaned:  {cleaned[:50]}...")

    # Exercise 2: Data augmentation
    print("\n2. Data augmentation...")

    # Load QA data for augmentation
    qa_examples = manager.load_from_json("data/sample_qa.json")

    # Apply augmentation
    augmentor = DataAugmentor()
    augmented_examples = augmentor.augment_qa_dataset(
        qa_examples[:3], augmentation_factor=0.5  # Use first 3 examples
    )

    print(f"   Original examples: {len(qa_examples[:3])}")
    print(f"   Augmented examples: {len(augmented_examples)}")
    print(f"   New examples created: {len(augmented_examples) - len(qa_examples[:3])}")

    # Show augmentation example
    if len(augmented_examples) > len(qa_examples[:3]):
        original_q = qa_examples[0].question
        augmented_q = augmented_examples[-1].question  # Last one should be augmented
        print(f"   Original:  {original_q}")
        print(f"   Augmented: {augmented_q}")

    # Exercise 3: Pipeline preprocessing
    print("\n3. Pipeline preprocessing...")

    # Create a multi-stage preprocessing pipeline
    def create_preprocessing_pipeline():
        """Create a multi-stage preprocessing pipeline"""

        # Stage 1: Basic cleaning
        stage1_config = PreprocessingConfig(
            remove_extra_whitespace=True, normalize_unicode=True, remove_html_tags=True
        )

        # Stage 2: Text normalization
        stage2_config = PreprocessingConfig(
            normalize_quotes=True, normalize_dashes=True, remove_extra_whitespace=True
        )

        # Stage 3: Quality filtering
        stage3_config = PreprocessingConfig(
            min_text_length=10, min_word_count=3, remove_duplicates=True
        )

        return [
            ("cleaning", DataPreprocessor(stage1_config)),
            ("normalization", DataPreprocessor(stage2_config)),
            ("filtering", DataPreprocessor(stage3_config)),
        ]

    # Apply pipeline
    pipeline = create_preprocessing_pipeline()
    current_examples = qa_examples.copy()

    for stage_name, preprocessor in pipeline:
        before_count = len(current_examples)
        current_examples = preprocessor.preprocess_dataset(
            current_examples, ["question", "answer"]
        )
        after_count = len(current_examples)
        print(f"   {stage_name}: {before_count} -> {after_count} examples")

    print(f"   Final pipeline result: {len(current_examples)} examples")

    # Exercise 4: Validation after preprocessing
    print("\n4. Validation after preprocessing...")

    # Validate the preprocessed data
    validator = DataValidator()
    quality_metrics = validator.validate_dataset_quality(
        current_examples, ["question", "answer"]
    )

    print(f"   Overall quality score: {quality_metrics['overall_quality_score']:.2%}")
    print(f"   Valid examples: {quality_metrics['valid_examples']}")

    # Show field quality
    for field, metrics in quality_metrics["field_quality"].items():
        avg_length = metrics["avg_length"]
        avg_words = metrics["avg_word_count"]
        print(f"   {field}: avg_length={avg_length:.1f}, avg_words={avg_words:.1f}")

    print("\n=== Advanced Preprocessing Complete ===")
    return current_examples


def solution_specialized_preprocessing():
    """
    Solution for specialized preprocessing tasks
    """
    print("\n=== Specialized Preprocessing ===\n")

    manager = DatasetManager()
    # Load QA examples for batch processing and quality assessment
    qa_examples = manager.load_from_json("data/sample_qa.json")

    # Exercise 1: RAG dataset preprocessing
    print("1. RAG dataset preprocessing...")

    # Load RAG data
    rag_examples = manager.load_from_jsonl("data/sample_rag.jsonl")

    # Configure RAG-specific preprocessing
    rag_config = PreprocessingConfig(
        remove_extra_whitespace=True,
        normalize_unicode=True,
        normalize_quotes=True,
        min_text_length=20,  # Longer minimum for context
        max_text_length=2000,  # Allow longer context
        remove_duplicates=True,
    )

    processed_rag = preprocess_rag_dataset(rag_examples, rag_config)

    print(f"   ✅ Processed {len(processed_rag)} RAG examples")

    # Analyze context lengths
    if processed_rag:
        context_lengths = [
            len(ex.context) for ex in processed_rag if hasattr(ex, "context")
        ]
        if context_lengths:
            avg_context_length = sum(context_lengths) / len(context_lengths)
            print(f"   Average context length: {avg_context_length:.1f} characters")

    # Exercise 2: Domain-specific cleaning
    print("\n2. Domain-specific cleaning...")

    # Create domain-specific examples
    technical_examples = [
        Example(
            question="What is API rate limiting?",
            answer="API rate limiting controls the number of requests per time period (e.g., 100 req/min).",
        ),
        Example(
            question="How does OAuth 2.0 work?",
            answer="OAuth 2.0 provides secure authorization via access tokens & refresh tokens.",
        ),
    ]

    # Configure for technical content
    technical_config = PreprocessingConfig(
        remove_extra_whitespace=True,
        normalize_unicode=True,
        normalize_quotes=True,
        remove_punctuation=False,  # Keep technical punctuation
        remove_numbers=False,  # Keep version numbers, etc.
        min_text_length=15,
    )

    technical_preprocessor = DataPreprocessor(technical_config)
    processed_technical = technical_preprocessor.preprocess_dataset(
        technical_examples, ["question", "answer"]
    )

    print(f"   ✅ Processed {len(processed_technical)} technical examples")

    # Exercise 3: Batch processing optimization
    print("\n3. Batch processing optimization...")

    # Simulate large dataset processing
    large_dataset = qa_examples * 10  # Simulate larger dataset

    # Process in batches
    batch_size = 50
    processed_batches = []

    config = PreprocessingConfig(
        remove_extra_whitespace=True,
        normalize_unicode=True,
        min_text_length=10,
        remove_duplicates=True,
    )

    preprocessor = DataPreprocessor(config)

    for i in range(0, len(large_dataset), batch_size):
        batch = large_dataset[i : i + batch_size]
        processed_batch = preprocessor.preprocess_dataset(batch, ["question", "answer"])
        processed_batches.extend(processed_batch)
        print(
            f"   Processed batch {i//batch_size + 1}: {len(processed_batch)} examples"
        )

    print(f"   Total processed: {len(processed_batches)} examples")

    # Exercise 4: Quality assessment after preprocessing
    print("\n4. Quality assessment after preprocessing...")

    # Compare quality before and after preprocessing
    validator = DataValidator()

    # Original quality
    original_quality = validator.validate_dataset_quality(
        qa_examples, ["question", "answer"]
    )

    # Processed quality
    processed_quality = validator.validate_dataset_quality(
        processed_batches[: len(qa_examples)], ["question", "answer"]
    )

    print(f"   Original quality: {original_quality['overall_quality_score']:.2%}")
    print(f"   Processed quality: {processed_quality['overall_quality_score']:.2%}")

    improvement = (
        processed_quality["overall_quality_score"]
        - original_quality["overall_quality_score"]
    )
    print(f"   Quality improvement: {improvement:+.2%}")

    print("\n=== Specialized Preprocessing Complete ===")


if __name__ == "__main__":
    # Run basic preprocessing solution
    preprocessed_data = solution_basic_preprocessing()

    # Run advanced preprocessing solution
    advanced_data = solution_advanced_preprocessing()

    # Run specialized preprocessing solution
    solution_specialized_preprocessing()

    print("\n🎉 All preprocessing exercises completed successfully!")
    print("\nKey takeaways:")
    print("- Configure preprocessing based on your specific use case")
    print("- Apply quality filters to remove problematic examples")
    print("- Use multi-stage pipelines for complex preprocessing")
    print("- Always validate data quality after preprocessing")
    print("- Consider domain-specific requirements when cleaning text")
    print("- Process large datasets in batches for efficiency")
