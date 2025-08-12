"""
Data Preprocessing Pipeline for DSPy Examples

This module provides comprehensive data preprocessing utilities for cleaning,
transforming, and augmenting DSPy Example datasets. It includes text cleaning,
normalization, augmentation, and quality improvement techniques.

Key Features:
- Text cleaning and normalization
- Data augmentation and synthesis
- Format standardization
- Quality improvement filters
- Batch processing capabilities
"""

import logging
import re
import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from dspy import Example

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing operations"""

    # Text cleaning options
    remove_extra_whitespace: bool = True
    normalize_unicode: bool = True
    remove_html_tags: bool = True
    remove_urls: bool = False
    remove_emails: bool = False
    remove_phone_numbers: bool = False

    # Text normalization options
    lowercase: bool = False
    remove_punctuation: bool = False
    remove_numbers: bool = False
    normalize_quotes: bool = True
    normalize_dashes: bool = True

    # Length filtering
    min_text_length: Optional[int] = None
    max_text_length: Optional[int] = None

    # Quality filtering
    min_word_count: Optional[int] = None
    max_word_count: Optional[int] = None
    remove_duplicates: bool = True

    # Language processing
    remove_non_ascii: bool = False
    remove_non_english: bool = False


class TextCleaner:
    """Text cleaning utilities for preprocessing"""

    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Remove HTML tags from text"""
        html_pattern = re.compile(r"<[^>]+>")
        return html_pattern.sub("", text)

    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text"""
        url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        return url_pattern.sub("", text)

    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text"""
        email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )
        return email_pattern.sub("", text)

    @staticmethod
    def remove_phone_numbers(text: str) -> str:
        """Remove phone numbers from text"""
        phone_pattern = re.compile(
            r"(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        )
        return phone_pattern.sub("", text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text"""
        # Replace multiple whitespace with single space
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace
        return text.strip()

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters"""
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def normalize_quotes(text: str) -> str:
        """Normalize different quote characters"""
        # Replace curly quotes with straight quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(""", "'").replace(""", "'")
        return text

    @staticmethod
    def normalize_dashes(text: str) -> str:
        """Normalize different dash characters"""
        # Replace em dash and en dash with regular dash
        text = text.replace("—", "-").replace("–", "-")
        return text

    @staticmethod
    def remove_punctuation(text: str) -> str:
        """Remove punctuation from text"""
        return text.translate(str.maketrans("", "", string.punctuation))

    @staticmethod
    def remove_numbers(text: str) -> str:
        """Remove numbers from text"""
        return re.sub(r"\d+", "", text)

    @staticmethod
    def remove_non_ascii(text: str) -> str:
        """Remove non-ASCII characters"""
        return "".join(char for char in text if ord(char) < 128)

    @staticmethod
    def remove_non_english(text: str) -> str:
        """Remove non-English characters (basic approach)"""
        # Keep only ASCII letters, numbers, and common punctuation
        pattern = re.compile(r'[^a-zA-Z0-9\s\.,!?;:\'"()-]')
        return pattern.sub("", text)


class DataPreprocessor:
    """
    Main data preprocessing pipeline for DSPy Examples

    Provides comprehensive preprocessing capabilities including cleaning,
    normalization, filtering, and augmentation.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor

        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.cleaner = TextCleaner()

    def clean_text(self, text: str) -> str:
        """
        Apply text cleaning operations

        Args:
            text: Input text to clean

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return str(text)

        # Apply cleaning operations based on config
        if self.config.remove_html_tags:
            text = self.cleaner.remove_html_tags(text)

        if self.config.remove_urls:
            text = self.cleaner.remove_urls(text)

        if self.config.remove_emails:
            text = self.cleaner.remove_emails(text)

        if self.config.remove_phone_numbers:
            text = self.cleaner.remove_phone_numbers(text)

        if self.config.normalize_unicode:
            text = self.cleaner.normalize_unicode(text)

        if self.config.normalize_quotes:
            text = self.cleaner.normalize_quotes(text)

        if self.config.normalize_dashes:
            text = self.cleaner.normalize_dashes(text)

        if self.config.remove_extra_whitespace:
            text = self.cleaner.normalize_whitespace(text)

        # Apply normalization operations
        if self.config.lowercase:
            text = text.lower()

        if self.config.remove_punctuation:
            text = self.cleaner.remove_punctuation(text)

        if self.config.remove_numbers:
            text = self.cleaner.remove_numbers(text)

        if self.config.remove_non_ascii:
            text = self.cleaner.remove_non_ascii(text)

        if self.config.remove_non_english:
            text = self.cleaner.remove_non_english(text)

        return text

    def filter_by_length(self, text: str) -> bool:
        """
        Check if text meets length requirements

        Args:
            text: Text to check

        Returns:
            True if text meets length requirements
        """
        text_length = len(text)

        if self.config.min_text_length and text_length < self.config.min_text_length:
            return False

        if self.config.max_text_length and text_length > self.config.max_text_length:
            return False

        return True

    def filter_by_word_count(self, text: str) -> bool:
        """
        Check if text meets word count requirements

        Args:
            text: Text to check

        Returns:
            True if text meets word count requirements
        """
        word_count = len(text.split())

        if self.config.min_word_count and word_count < self.config.min_word_count:
            return False

        if self.config.max_word_count and word_count > self.config.max_word_count:
            return False

        return True

    def preprocess_example(
        self, example: Example, text_fields: List[str]
    ) -> Optional[Example]:
        """
        Preprocess a single DSPy Example

        Args:
            example: DSPy Example to preprocess
            text_fields: List of field names containing text to preprocess

        Returns:
            Preprocessed example or None if filtered out
        """
        try:
            # Create a copy of the example data
            example_data = {}
            for key, value in example.__dict__.items():
                if not key.startswith("_"):
                    example_data[key] = value

            # Process text fields
            for field in text_fields:
                if field in example_data and example_data[field] is not None:
                    original_text = str(example_data[field])
                    cleaned_text = self.clean_text(original_text)

                    # Apply filters
                    if not self.filter_by_length(cleaned_text):
                        logger.debug(f"Filtered out example due to length: {field}")
                        return None

                    if not self.filter_by_word_count(cleaned_text):
                        logger.debug(f"Filtered out example due to word count: {field}")
                        return None

                    example_data[field] = cleaned_text

            return Example(**example_data)

        except Exception as e:
            logger.warning(f"Error preprocessing example: {e}")
            return None

    def preprocess_dataset(
        self, examples: List[Example], text_fields: List[str]
    ) -> List[Example]:
        """
        Preprocess a dataset of DSPy Examples

        Args:
            examples: List of DSPy Examples to preprocess
            text_fields: List of field names containing text to preprocess

        Returns:
            List of preprocessed examples
        """
        preprocessed = []
        filtered_count = 0

        for i, example in enumerate(examples):
            processed_example = self.preprocess_example(example, text_fields)

            if processed_example is not None:
                preprocessed.append(processed_example)
            else:
                filtered_count += 1

        # Remove duplicates if configured
        if self.config.remove_duplicates:
            original_count = len(preprocessed)
            preprocessed = self.remove_duplicates(preprocessed, text_fields)
            duplicate_count = original_count - len(preprocessed)
            logger.info(f"Removed {duplicate_count} duplicate examples")

        logger.info(
            f"Preprocessed {len(examples)} examples: "
            f"{len(preprocessed)} kept, {filtered_count} filtered out"
        )

        return preprocessed

    def remove_duplicates(
        self, examples: List[Example], text_fields: List[str]
    ) -> List[Example]:
        """
        Remove duplicate examples based on text fields

        Args:
            examples: List of DSPy Examples
            text_fields: Fields to use for duplicate detection

        Returns:
            List of unique examples
        """
        seen_texts = set()
        unique_examples = []

        for example in examples:
            # Create a signature from text fields
            signature_parts = []
            for field in text_fields:
                if hasattr(example, field):
                    value = getattr(example, field)
                    if value is not None:
                        signature_parts.append(str(value).strip().lower())

            signature = "|".join(signature_parts)

            if signature not in seen_texts:
                seen_texts.add(signature)
                unique_examples.append(example)

        return unique_examples


class DataAugmentor:
    """Data augmentation utilities for DSPy Examples"""

    def __init__(self):
        """Initialize the data augmentor"""
        pass

    def paraphrase_text(self, text: str, num_variations: int = 1) -> List[str]:
        """
        Generate paraphrases of text (placeholder implementation)

        Args:
            text: Original text
            num_variations: Number of paraphrases to generate

        Returns:
            List of paraphrased texts
        """
        # This is a placeholder - in practice, you would use a paraphrasing model
        variations = []

        # Simple word substitutions as example
        substitutions = {
            "good": ["excellent", "great", "fine"],
            "bad": ["poor", "terrible", "awful"],
            "big": ["large", "huge", "massive"],
            "small": ["tiny", "little", "compact"],
        }

        for i in range(num_variations):
            modified_text = text
            for original, replacements in substitutions.items():
                if original in modified_text.lower():
                    replacement = replacements[i % len(replacements)]
                    modified_text = re.sub(
                        r"\b" + re.escape(original) + r"\b",
                        replacement,
                        modified_text,
                        flags=re.IGNORECASE,
                    )
            variations.append(modified_text)

        return variations

    def augment_qa_dataset(
        self, examples: List[Example], augmentation_factor: float = 0.5
    ) -> List[Example]:
        """
        Augment a question-answer dataset

        Args:
            examples: Original QA examples
            augmentation_factor: Fraction of examples to augment

        Returns:
            Augmented dataset
        """
        augmented = examples.copy()
        num_to_augment = int(len(examples) * augmentation_factor)

        # Select random examples to augment
        import random

        examples_to_augment = random.sample(
            examples, min(num_to_augment, len(examples))
        )

        for example in examples_to_augment:
            if hasattr(example, "question") and hasattr(example, "answer"):
                # Generate question variations
                question_variations = self.paraphrase_text(example.question, 1)

                for variation in question_variations:
                    # Create new example with paraphrased question
                    augmented_data = {}
                    for key, value in example.__dict__.items():
                        if not key.startswith("_"):
                            augmented_data[key] = value

                    augmented_data["question"] = variation
                    augmented.append(Example(**augmented_data))

        logger.info(
            f"Augmented dataset from {len(examples)} to {len(augmented)} examples"
        )
        return augmented


class DataValidator:
    """Data validation utilities for preprocessed datasets"""

    def __init__(self):
        """Initialize the data validator"""
        pass

    def validate_text_quality(self, text: str) -> Dict[str, Any]:
        """
        Validate text quality metrics

        Args:
            text: Text to validate

        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            "length": len(text),
            "word_count": len(text.split()),
            "sentence_count": len(re.split(r"[.!?]+", text)),
            "avg_word_length": 0,
            "has_punctuation": bool(re.search(r"[.!?]", text)),
            "has_uppercase": bool(re.search(r"[A-Z]", text)),
            "has_lowercase": bool(re.search(r"[a-z]", text)),
            "repetition_ratio": 0,
        }

        words = text.split()
        if words:
            metrics["avg_word_length"] = sum(len(word) for word in words) / len(words)

            # Calculate repetition ratio
            unique_words = set(words)
            metrics["repetition_ratio"] = 1 - (len(unique_words) / len(words))

        return metrics

    def validate_dataset_quality(
        self, examples: List[Example], text_fields: List[str]
    ) -> Dict[str, Any]:
        """
        Validate overall dataset quality

        Args:
            examples: List of DSPy Examples
            text_fields: Text fields to validate

        Returns:
            Dictionary with dataset quality metrics
        """
        if not examples:
            return {"error": "Empty dataset"}

        quality_metrics = {
            "total_examples": len(examples),
            "field_quality": {},
            "overall_quality_score": 0,
        }

        for field in text_fields:
            field_metrics = {
                "valid_count": 0,
                "avg_length": 0,
                "avg_word_count": 0,
                "quality_issues": [],
            }

            lengths = []
            word_counts = []

            for example in examples:
                if hasattr(example, field):
                    text = getattr(example, field)
                    if text and isinstance(text, str):
                        field_metrics["valid_count"] += 1
                        text_quality = self.validate_text_quality(text)

                        lengths.append(text_quality["length"])
                        word_counts.append(text_quality["word_count"])

                        # Check for quality issues
                        if text_quality["length"] < 10:
                            field_metrics["quality_issues"].append("very_short_text")
                        if text_quality["repetition_ratio"] > 0.5:
                            field_metrics["quality_issues"].append("high_repetition")
                        if not text_quality["has_punctuation"]:
                            field_metrics["quality_issues"].append("no_punctuation")

            if lengths:
                field_metrics["avg_length"] = sum(lengths) / len(lengths)
                field_metrics["avg_word_count"] = sum(word_counts) / len(word_counts)

            quality_metrics["field_quality"][field] = field_metrics

        # Calculate overall quality score
        total_valid = sum(
            metrics["valid_count"]
            for metrics in quality_metrics["field_quality"].values()
        )
        total_possible = len(examples) * len(text_fields)
        quality_metrics["overall_quality_score"] = (
            total_valid / total_possible if total_possible > 0 else 0
        )

        return quality_metrics


# Utility functions for common preprocessing tasks


def create_preprocessing_pipeline(config: PreprocessingConfig) -> DataPreprocessor:
    """
    Create a preprocessing pipeline with the given configuration

    Args:
        config: Preprocessing configuration

    Returns:
        Configured DataPreprocessor
    """
    return DataPreprocessor(config)


def preprocess_qa_dataset(
    examples: List[Example], config: Optional[PreprocessingConfig] = None
) -> List[Example]:
    """
    Preprocess a question-answer dataset

    Args:
        examples: List of QA examples
        config: Preprocessing configuration

    Returns:
        Preprocessed examples
    """
    preprocessor = DataPreprocessor(config)
    return preprocessor.preprocess_dataset(examples, ["question", "answer"])


def preprocess_classification_dataset(
    examples: List[Example], config: Optional[PreprocessingConfig] = None
) -> List[Example]:
    """
    Preprocess a classification dataset

    Args:
        examples: List of classification examples
        config: Preprocessing configuration

    Returns:
        Preprocessed examples
    """
    preprocessor = DataPreprocessor(config)
    return preprocessor.preprocess_dataset(examples, ["text"])


def preprocess_rag_dataset(
    examples: List[Example], config: Optional[PreprocessingConfig] = None
) -> List[Example]:
    """
    Preprocess a RAG dataset

    Args:
        examples: List of RAG examples
        config: Preprocessing configuration

    Returns:
        Preprocessed examples
    """
    preprocessor = DataPreprocessor(config)
    return preprocessor.preprocess_dataset(examples, ["question", "context", "answer"])


if __name__ == "__main__":
    # Example usage
    from dataset_management import DatasetManager

    # Load sample data
    manager = DatasetManager()
    examples = manager.load_from_json("data/sample_qa.json")

    # Create preprocessing configuration
    config = PreprocessingConfig(
        remove_extra_whitespace=True,
        normalize_unicode=True,
        normalize_quotes=True,
        min_text_length=10,
        min_word_count=3,
        remove_duplicates=True,
    )

    # Preprocess the dataset
    preprocessed = preprocess_qa_dataset(examples, config)

    # Validate quality
    validator = DataValidator()
    quality = validator.validate_dataset_quality(preprocessed, ["question", "answer"])

    print(f"Original examples: {len(examples)}")
    print(f"Preprocessed examples: {len(preprocessed)}")
    print(f"Overall quality score: {quality['overall_quality_score']:.2%}")

    # Augment dataset
    augmentor = DataAugmentor()
    augmented = augmentor.augment_qa_dataset(preprocessed, augmentation_factor=0.3)
    print(f"Augmented examples: {len(augmented)}")
