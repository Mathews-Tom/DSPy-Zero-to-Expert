"""
Solution 01: Basic Dataset Loading

This solution demonstrates how to load datasets from various formats
and work with DSPy Example objects effectively.

Learning Objectives:
- Load datasets from JSON, JSONL, and CSV formats
- Create and manipulate DSPy Example objects
- Handle loading errors gracefully
- Validate loaded data
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from dataset_management import (
    DatasetManager,
    create_classification_examples,
    create_qa_examples,
)
from dspy import Example  # pylint: disable=import-error


def solution_basic_loading():
    """
    Solution for basic dataset loading exercises
    """
    print("=== Solution 01: Basic Dataset Loading ===\n")

    # Initialize dataset manager
    manager = DatasetManager()

    # Exercise 1: Load JSON dataset
    print("1. Loading JSON dataset...")
    try:
        json_examples = manager.load_from_json("data/sample_qa.json")
        print(f"   ✅ Loaded {len(json_examples)} examples from JSON")

        # Display first example
        if json_examples:
            first_example = json_examples[0]
            print(f"   First example: {first_example.question[:50]}...")
            print(f"   Answer: {first_example.answer[:50]}...")
    except Exception as e:
        print(f"   ❌ Error loading JSON: {e}")

    # Exercise 2: Load JSONL dataset
    print("\n2. Loading JSONL dataset...")
    try:
        jsonl_examples = manager.load_from_jsonl("data/sample_rag.jsonl")
        print(f"   ✅ Loaded {len(jsonl_examples)} examples from JSONL")

        # Display first example fields
        if jsonl_examples:
            first_example = jsonl_examples[0]
            fields = [
                key for key in first_example.__dict__.keys() if not key.startswith("_")
            ]
            print(f"   Fields in first example: {fields}")
    except Exception as e:
        print(f"   ❌ Error loading JSONL: {e}")

    # Exercise 3: Load CSV dataset
    print("\n3. Loading CSV dataset...")
    try:
        csv_examples = manager.load_from_csv(
            "data/sample_classification.csv", "text", "label"
        )
        print(f"   ✅ Loaded {len(csv_examples)} examples from CSV")

        # Show label distribution
        if csv_examples:
            labels = [ex.label for ex in csv_examples if hasattr(ex, "label")]
            from collections import Counter

            label_counts = Counter(labels)
            print(f"   Label distribution: {dict(label_counts)}")
    except Exception as e:
        print(f"   ❌ Error loading CSV: {e}")

    # Exercise 4: Create examples programmatically
    print("\n4. Creating examples programmatically...")

    # Create QA examples
    questions = [
        "What is the capital of France?",
        "How do you create a list in Python?",
        "What is machine learning?",
    ]
    answers = [
        "The capital of France is Paris.",
        "You can create a list using square brackets: my_list = [1, 2, 3]",
        "Machine learning is a subset of AI that learns from data.",
    ]

    qa_examples = create_qa_examples(questions, answers)
    print(f"   ✅ Created {len(qa_examples)} QA examples")

    # Create classification examples
    texts = [
        "This product is amazing!",
        "Terrible quality, very disappointed.",
        "It's okay, nothing special.",
    ]
    labels = ["positive", "negative", "neutral"]

    classification_examples = create_classification_examples(texts, labels)
    print(f"   ✅ Created {len(classification_examples)} classification examples")

    # Exercise 5: Save examples to different formats
    print("\n5. Saving examples to different formats...")

    # Save QA examples as JSON
    success = manager.save_to_json(qa_examples, "output_qa_examples.json")
    if success:
        print("   ✅ Saved QA examples to JSON")

    # Save classification examples as JSONL
    success = manager.save_to_jsonl(
        classification_examples, "output_classification_examples.jsonl"
    )
    if success:
        print("   ✅ Saved classification examples to JSONL")

    # Exercise 6: Validate loaded data
    print("\n6. Validating loaded data...")

    # Validate JSON examples
    if "json_examples" in locals():
        valid_examples, errors = manager.validate_examples(
            json_examples, required_fields=["question", "answer"]
        )
        print(
            f"   JSON validation: {len(valid_examples)}/{len(json_examples)} valid examples"
        )
        if errors:
            print(f"   First error: {errors[0]}")

    # Validate CSV examples
    if "csv_examples" in locals():
        valid_examples, errors = manager.validate_examples(
            csv_examples, required_fields=["text", "label"]
        )
        print(
            f"   CSV validation: {len(valid_examples)}/{len(csv_examples)} valid examples"
        )

    # Exercise 7: Handle missing files gracefully
    print("\n7. Handling missing files...")

    # Try to load non-existent file
    missing_examples = manager.load_from_json("non_existent_file.json")
    print(
        f"   Loading missing file returned {len(missing_examples)} examples (expected 0)"
    )

    # Exercise 8: Work with Example object properties
    print("\n8. Working with Example object properties...")

    if "json_examples" in locals() and json_examples:
        example = json_examples[0]

        # Access properties
        print(f"   Example has question: {hasattr(example, 'question')}")
        print(f"   Example has answer: {hasattr(example, 'answer')}")
        print(f"   Example has category: {hasattr(example, 'category')}")

        # Get all properties
        properties = [key for key in example.__dict__.keys() if not key.startswith("_")]
        print(f"   All properties: {properties}")

        # Modify example
        if hasattr(example, "category"):
            print(f"   Original category: {example.category}")
            # Note: DSPy Examples are typically immutable, so we create a new one
            modified_data = {key: getattr(example, key) for key in properties}
            modified_data["category"] = "modified"
            modified_example = Example(**modified_data)
            print(f"   Modified category: {modified_example.category}")

    print("\n=== Solution 01 Complete ===")
    return True


def demonstrate_advanced_loading():
    """
    Demonstrate advanced loading techniques
    """
    print("\n=== Advanced Loading Techniques ===\n")

    manager = DatasetManager()

    # 1. Batch loading multiple files
    print("1. Batch loading multiple files...")
    all_examples = []

    file_paths = [("data/sample_qa.json", "json"), ("data/sample_rag.jsonl", "jsonl")]

    for file_path, file_type in file_paths:
        try:
            if file_type == "json":
                examples = manager.load_from_json(file_path)
            elif file_type == "jsonl":
                examples = manager.load_from_jsonl(file_path)
            else:
                continue

            all_examples.extend(examples)
            print(f"   ✅ Loaded {len(examples)} examples from {file_path}")
        except Exception as e:
            print(f"   ❌ Error loading {file_path}: {e}")

    print(f"   Total examples loaded: {len(all_examples)}")

    # 2. Loading with custom validation
    print("\n2. Loading with custom validation...")

    def custom_validator(example):
        """Custom validation function"""
        # Check if example has required fields and they're not empty
        if hasattr(example, "question") and hasattr(example, "answer"):
            question = getattr(example, "question")
            answer = getattr(example, "answer")
            if isinstance(question, str) and isinstance(answer, str):
                if len(question.strip()) > 5 and len(answer.strip()) > 5:
                    return True
        return False

    # Load and filter examples
    raw_examples = manager.load_from_json("data/sample_qa.json")
    valid_examples = [ex for ex in raw_examples if custom_validator(ex)]

    print(f"   Raw examples: {len(raw_examples)}")
    print(f"   Valid examples: {len(valid_examples)}")
    print(f"   Filtered out: {len(raw_examples) - len(valid_examples)}")

    # 3. Loading with data transformation
    print("\n3. Loading with data transformation...")

    def transform_example(example):
        """Transform example during loading"""
        # Create new example with transformed data
        transformed_data = {}

        for key, value in example.__dict__.items():
            if not key.startswith("_"):
                if isinstance(value, str):
                    # Clean and normalize text
                    transformed_data[key] = value.strip().replace("\n", " ")
                else:
                    transformed_data[key] = value

        # Add metadata
        transformed_data["loaded_at"] = "2024-01-01"
        transformed_data["source"] = "sample_data"

        return Example(**transformed_data)

    # Load and transform examples
    raw_examples = manager.load_from_json("data/sample_qa.json")
    transformed_examples = [transform_example(ex) for ex in raw_examples]

    print(f"   Transformed {len(transformed_examples)} examples")
    if transformed_examples:
        first_transformed = transformed_examples[0]
        print(
            f"   Added metadata: loaded_at={getattr(first_transformed, 'loaded_at', 'N/A')}"
        )

    print("\n=== Advanced Loading Complete ===")


if __name__ == "__main__":
    # Run the basic solution
    solution_basic_loading()

    # Run advanced demonstrations
    demonstrate_advanced_loading()

    print("\n🎉 All exercises completed successfully!")
    print("\nKey takeaways:")
    print("- DSPy Examples are flexible containers for structured data")
    print("- Always validate data after loading")
    print("- Handle errors gracefully when loading files")
    print("- Use appropriate formats for different use cases")
    print("- Transform data during loading when needed")
