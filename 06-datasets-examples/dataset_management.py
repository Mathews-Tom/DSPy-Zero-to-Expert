"""
Dataset Management System for DSPy Examples

This module provides comprehensive utilities for managing DSPy Example objects,
including loading, preprocessing, validation, and quality checking of datasets
used in DSPy applications.

Key Features:
- Dataset loading from various formats (JSON, CSV, JSONL)
- DSPy Example object creation and validation
- Dataset splitting and sampling utilities
- Quality checking and validation metrics
- Dataset transformation and preprocessing
"""

import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dspy import Example  # pylint: disable=import-error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics about a dataset"""

    total_examples: int
    field_counts: dict[str, int]
    field_types: dict[str, str]
    missing_values: dict[str, int]
    unique_values: dict[str, int]
    avg_text_length: dict[str, float]


class DatasetManager:
    """
    Comprehensive dataset management system for DSPy Examples

    Handles loading, validation, splitting, and quality assessment
    of datasets for DSPy applications.
    """

    def __init__(self, data_dir: str | None = None):
        """
        Initialize the dataset manager

        Args:
            data_dir: Directory containing dataset files
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.data_dir.mkdir(exist_ok=True)

    def load_from_json(self, file_path: str) -> list[Example]:
        """
        Load dataset from JSON file

        Args:
            file_path: Path to JSON file

        Returns:
            list of DSPy Example objects
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            examples = []
            for item in data:
                if isinstance(item, dict):
                    examples.append(Example(**item))
                else:
                    logger.warning("Skipping non-dict item: %s", item)

            logger.info("Loaded %d examples from %s", len(examples), file_path)
            return examples

        except Exception as e:
            logger.error("Error loading JSON file %s: %s", file_path, e)
            return []

    def load_from_jsonl(self, file_path: str) -> list[Example]:
        """
        Load dataset from JSONL file

        Args:
            file_path: Path to JSONL file

        Returns:
            list of DSPy Example objects
        """
        try:
            examples = []
            with open(file_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            examples.append(Example(**item))
                        else:
                            logger.warning("Line %d: Skipping non-dict item", line_num)
                    except json.JSONDecodeError as e:
                        logger.warning("Line %d: Invalid JSON - %s", line_num, e)

            logger.info("Loaded %d examples from %s", len(examples), file_path)
            return examples

        except Exception as e:
            logger.error("Error loading JSONL file %s: %s", file_path, e)
            return []

    def load_from_csv(
        self, file_path: str, input_col: str, output_col: str
    ) -> list[Example]:
        """
        Load dataset from CSV file

        Args:
            file_path: Path to CSV file
            input_col: Column name for input data
            output_col: Column name for output data

        Returns:
            list of DSPy Example objects
        """
        try:
            examples = []
            with open(file_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader, 1):
                    try:
                        if input_col in row and output_col in row:
                            example_data = {
                                input_col: row[input_col],
                                output_col: row[output_col],
                            }
                            # Add any additional columns
                            for key, value in row.items():
                                if key not in [input_col, output_col]:
                                    example_data[key] = value

                            examples.append(Example(**example_data))
                        else:
                            logger.warning("Row %d: Missing required columns", row_num)
                    except Exception as e:
                        logger.warning(
                            "Row %d: Error creating example - %s", row_num, e
                        )

            logger.info("Loaded %d examples from %s", len(examples), file_path)
            return examples

        except Exception as e:
            logger.error("Error loading CSV file %s: %s", file_path, e)
            return []

    def save_to_json(self, examples: list[Example], file_path: str) -> bool:
        """
        Save examples to JSON file

        Args:
            examples: list of DSPy Example objects
            file_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            data = []
            for example in examples:
                # Convert Example to dict
                example_dict = {}
                for key, value in example.__dict__.items():
                    if not key.startswith("_"):
                        example_dict[key] = value
                data.append(example_dict)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info("Saved %d examples to %s", len(examples), file_path)
            return True

        except Exception as e:
            logger.error("Error saving to JSON file %s: %s", file_path, e)
            return False

    def save_to_jsonl(self, examples: list[Example], file_path: str) -> bool:
        """
        Save examples to JSONL file

        Args:
            examples: list of DSPy Example objects
            file_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for example in examples:
                    # Convert Example to dict
                    example_dict = {}
                    for key, value in example.__dict__.items():
                        if not key.startswith("_"):
                            example_dict[key] = value
                    f.write(json.dumps(example_dict, ensure_ascii=False) + "\n")

            logger.info("Saved %d examples to %s", len(examples), file_path)
            return True

        except Exception as e:
            logger.error("Error saving to JSONL file %s: %s", file_path, e)
            return False

    def validate_examples(
        self, examples: list[Example], required_fields: list[str]
    ) -> tuple[list[Example], list[str]]:
        """
        Validate examples and return valid ones with error messages

        Args:
            examples: list of DSPy Example objects
            required_fields: list of required field names

        Returns:
            tuple of (valid_examples, error_messages)
        """
        valid_examples = []
        errors = []

        for i, example in enumerate(examples):
            try:
                # Check required fields
                missing_fields = []
                for field in required_fields:
                    if not hasattr(example, field) or getattr(example, field) is None:
                        missing_fields.append(field)

                if missing_fields:
                    errors.append(f"Example {i}: Missing fields {missing_fields}")
                    continue

                # Check for empty string values in required fields
                empty_fields = []
                for field in required_fields:
                    value = getattr(example, field)
                    if isinstance(value, str) and not value.strip():
                        empty_fields.append(field)

                if empty_fields:
                    errors.append(f"Example {i}: Empty fields {empty_fields}")
                    continue

                valid_examples.append(example)

            except Exception as e:
                errors.append(f"Example {i}: Validation error - {e}")

        logger.info("Validated %d/%d examples", len(valid_examples), len(examples))
        if errors:
            logger.warning("Found %d validation errors", len(errors))

        return valid_examples, errors

    def split_dataset(
        self,
        examples: list[Example],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> dict[str, list[Example]]:
        """
        Split dataset into train/validation/test sets

        Args:
            examples: list of DSPy Example objects
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        # Shuffle examples
        random.seed(random_seed)
        shuffled_examples = examples.copy()
        random.shuffle(shuffled_examples)

        # Calculate split indices
        total_examples = len(shuffled_examples)
        train_end = int(total_examples * train_ratio)
        val_end = train_end + int(total_examples * val_ratio)

        # Split the data
        splits = {
            "train": shuffled_examples[:train_end],
            "val": shuffled_examples[train_end:val_end],
            "test": shuffled_examples[val_end:],
        }

        logger.info(
            "Split dataset: train=%d, val=%d, test=%d",
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )

        return splits

    def sample_examples(
        self, examples: list[Example], n_samples: int, random_seed: int = 42
    ) -> list[Example]:
        """
        Sample n examples from the dataset

        Args:
            examples: list of DSPy Example objects
            n_samples: Number of samples to return
            random_seed: Random seed for reproducibility

        Returns:
            list of sampled examples
        """
        if n_samples >= len(examples):
            logger.warning(
                "Requested %d samples but only %d available", n_samples, len(examples)
            )
            return examples.copy()

        random.seed(random_seed)
        sampled = random.sample(examples, n_samples)

        logger.info("Sampled %d examples from %d", len(sampled), len(examples))
        return sampled

    def get_dataset_stats(self, examples: list[Example]) -> DatasetStats:
        """
        Calculate comprehensive statistics about the dataset

        Args:
            examples: list of DSPy Example objects

        Returns:
            DatasetStats object with statistics
        """
        if not examples:
            return DatasetStats(0, {}, {}, {}, {}, {})

        # Collect all fields
        all_fields = set()
        for example in examples:
            all_fields.update(example.__dict__.keys())
        all_fields = {f for f in all_fields if not f.startswith("_")}

        # Initialize counters
        field_counts = {field: 0 for field in all_fields}
        field_types = {}
        missing_values = {field: 0 for field in all_fields}
        unique_values = {field: set() for field in all_fields}
        text_lengths = {field: [] for field in all_fields}

        # Analyze each example
        for example in examples:
            for field in all_fields:
                if hasattr(example, field):
                    value = getattr(example, field)
                    if value is not None:
                        field_counts[field] += 1

                        # Track field type
                        field_type = type(value).__name__
                        if field not in field_types:
                            field_types[field] = field_type
                        elif field_types[field] != field_type:
                            field_types[field] = "mixed"

                        # Track unique values (limit to avoid memory issues)
                        if len(unique_values[field]) < 1000:
                            unique_values[field].add(str(value))

                        # Track text length for string fields
                        if isinstance(value, str):
                            text_lengths[field].append(len(value))
                    else:
                        missing_values[field] += 1
                else:
                    missing_values[field] += 1

        # Calculate average text lengths
        avg_text_length = {}
        for field, lengths in text_lengths.items():
            if lengths:
                avg_text_length[field] = sum(lengths) / len(lengths)
            else:
                avg_text_length[field] = 0.0

        # Convert unique values to counts
        unique_counts = {field: len(values) for field, values in unique_values.items()}

        return DatasetStats(
            total_examples=len(examples),
            field_counts=field_counts,
            field_types=field_types,
            missing_values=missing_values,
            unique_values=unique_counts,
            avg_text_length=avg_text_length,
        )

    def quality_check(
        self, examples: list[Example], required_fields: list[str]
    ) -> dict[str, Any]:
        """
        Perform comprehensive quality check on dataset

        Args:
            examples: list of DSPy Example objects
            required_fields: list of required field names

        Returns:
            Dictionary with quality metrics and issues
        """
        stats = self.get_dataset_stats(examples)
        valid_examples, validation_errors = self.validate_examples(
            examples, required_fields
        )

        # Calculate quality metrics
        completeness = {}
        for field in required_fields:
            if field in stats.field_counts:
                completeness[field] = stats.field_counts[field] / stats.total_examples
            else:
                completeness[field] = 0.0

        # Identify potential issues
        issues = []

        # Check for high missing value rates
        for field in required_fields:
            missing_rate = stats.missing_values.get(field, 0) / stats.total_examples
            if missing_rate > 0.1:  # More than 10% missing
                issues.append(f"High missing rate for {field}: {missing_rate:.2%}")

        # Check for very short text fields
        for field in required_fields:
            if field in stats.avg_text_length:
                avg_length = stats.avg_text_length[field]
                if avg_length < 10:  # Very short text
                    issues.append(
                        f"Very short average text length for {field}: {avg_length:.1f}"
                    )

        # Check for low diversity
        for field in required_fields:
            if field in stats.unique_values:
                unique_ratio = stats.unique_values[field] / stats.total_examples
                if unique_ratio < 0.1:  # Less than 10% unique values
                    issues.append(
                        f"Low diversity for {field}: {unique_ratio:.2%} unique"
                    )

        return {
            "stats": stats,
            "valid_examples": len(valid_examples),
            "validation_errors": validation_errors,
            "completeness": completeness,
            "issues": issues,
            "overall_quality": len(valid_examples) / len(examples) if examples else 0.0,
        }


# Utility functions for common dataset operations


def create_qa_examples(questions: list[str], answers: list[str]) -> list[Example]:
    """
    Create DSPy Examples from question-answer pairs

    Args:
        questions: list of questions
        answers: list of corresponding answers

    Returns:
        list of DSPy Example objects
    """
    if len(questions) != len(answers):
        raise ValueError("Questions and answers must have the same length")

    examples = []
    for q, a in zip(questions, answers):
        examples.append(Example(question=q, answer=a))

    return examples


def create_classification_examples(
    texts: list[str], labels: list[str]
) -> list[Example]:
    """
    Create DSPy Examples for classification tasks

    Args:
        texts: list of input texts
        labels: list of corresponding labels

    Returns:
        list of DSPy Example objects
    """
    if len(texts) != len(labels):
        raise ValueError("Texts and labels must have the same length")

    examples = []
    for text, label in zip(texts, labels):
        examples.append(Example(text=text, label=label))

    return examples


def merge_datasets(*datasets: list[Example]) -> list[Example]:
    """
    Merge multiple datasets into one

    Args:
        *datasets: Variable number of dataset lists

    Returns:
        Combined list of examples
    """
    merged = []
    for dataset in datasets:
        merged.extend(dataset)

    logger.info("Merged %d datasets into %d examples", len(datasets), len(merged))
    return merged


if __name__ == "__main__":
    # Example usage
    manager = DatasetManager()

    # Create sample data
    sample_examples = [
        Example(question="What is Python?", answer="A programming language"),
        Example(
            question="What is DSPy?",
            answer="A framework for programming with language models",
        ),
        Example(
            question="What is machine learning?",
            answer="A subset of AI that learns from data",
        ),
    ]

    # Save and load examples
    manager.save_to_json(sample_examples, "sample_dataset.json")
    loaded_examples = manager.load_from_json("sample_dataset.json")

    # Get statistics
    stats = manager.get_dataset_stats(loaded_examples)
    print(f"Dataset has {stats.total_examples} examples")
    print(f"Fields: {list(stats.field_counts.keys())}")

    # Quality check
    quality = manager.quality_check(loaded_examples, ["question", "answer"])
    print(f"Overall quality: {quality['overall_quality']:.2%}")

    # Split dataset
    splits = manager.split_dataset(loaded_examples)
    print(
        f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}"
    )
