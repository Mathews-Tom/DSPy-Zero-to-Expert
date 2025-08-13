#!/usr/bin/env python3
"""
DSPy Learning Framework - Module 1: Foundations Solutions

This script contains complete solutions for all exercises in Module 1: DSPy Foundations.
Each solution demonstrates core DSPy concepts including signatures, basic modules,
and fundamental patterns.

Learning Objectives Covered:
- Understanding DSPy signatures and their components
- Creating and using basic DSPy modules
- Working with input and output fields
- Implementing simple prediction tasks
- Basic error handling and validation

Author: DSPy Learning Framework
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import dspy
from utilities.dspy_helpers import setup_dspy_environment, validate_signature_output

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicSignature(dspy.Signature):
    """
    Exercise 1.1: Basic Signature Creation

    This signature demonstrates the fundamental structure of a DSPy signature
    with input and output fields, proper descriptions, and type annotations.
    """

    question: str = dspy.InputField(desc="Question to be answered")
    answer: str = dspy.OutputField(desc="Answer to the question")


class DetailedQASignature(dspy.Signature):
    """
    Exercise 1.2: Enhanced Question-Answering Signature

    This signature extends the basic QA pattern with additional context
    and confidence scoring for more sophisticated question answering.
    """

    question: str = dspy.InputField(desc="Question to be answered")
    context: str = dspy.InputField(desc="Context information for answering")
    answer: str = dspy.OutputField(desc="Detailed answer based on context")
    confidence: float = dspy.OutputField(desc="Confidence score (0-1) for the answer")


class TextClassificationSignature(dspy.Signature):
    """
    Exercise 1.3: Text Classification Signature

    This signature demonstrates classification tasks with multiple
    possible outputs and reasoning chains.
    """

    text: str = dspy.InputField(desc="Text to classify")
    categories: str = dspy.InputField(desc="Available categories (comma-separated)")
    category: str = dspy.OutputField(desc="Predicted category")
    reasoning: str = dspy.OutputField(desc="Reasoning for the classification")


class SentimentAnalysisSignature(dspy.Signature):
    """
    Exercise 1.4: Sentiment Analysis Signature

    This signature shows how to structure tasks that require
    both categorical and numerical outputs.
    """

    text: str = dspy.InputField(desc="Text to analyze for sentiment")
    sentiment: str = dspy.OutputField(
        desc="Sentiment label (positive/negative/neutral)"
    )
    score: float = dspy.OutputField(desc="Sentiment score (-1 to 1)")
    keywords: str = dspy.OutputField(desc="Key words that influenced the sentiment")


class TextSummarizationSignature(dspy.Signature):
    """
    Exercise 1.5: Text Summarization Signature

    This signature demonstrates how to handle variable-length inputs
    and outputs with specific formatting requirements.
    """

    text: str = dspy.InputField(desc="Text to summarize")
    max_length: int = dspy.InputField(desc="Maximum length of summary in words")
    summary: str = dspy.OutputField(desc="Concise summary of the text")
    key_points: str = dspy.OutputField(desc="Key points from the text (bullet points)")


def exercise_1_1_basic_signature():
    """
    Exercise 1.1: Create and test a basic DSPy signature

    This exercise demonstrates:
    - Creating a simple signature with input and output fields
    - Using the signature with a Predict module
    - Basic validation of outputs
    """
    print("\n" + "=" * 60)
    print("Exercise 1.1: Basic Signature Creation")
    print("=" * 60)

    try:
        # Create a Predict module with the basic signature
        predictor = dspy.Predict(BasicSignature)

        # Test with a simple question
        test_question = "What is the capital of France?"
        result = predictor(question=test_question)

        print(f"Question: {test_question}")
        print(f"Answer: {result.answer}")

        # Validate the result
        validation = validate_signature_output(
            BasicSignature,
            {"question": test_question},
            validation_rules={
                "answer": lambda x: len(x.strip()) > 0  # Answer should not be empty
            },
        )

        print(f"Validation: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}")

        return {
            "exercise": "1.1",
            "success": True,
            "result": result,
            "validation": validation,
        }

    except Exception as e:
        logger.error(f"Exercise 1.1 failed: {e}")
        return {"exercise": "1.1", "success": False, "error": str(e)}


def exercise_1_2_enhanced_qa():
    """
    Exercise 1.2: Enhanced Question-Answering with Context

    This exercise demonstrates:
    - Using multiple input fields
    - Incorporating context into reasoning
    - Working with confidence scores
    """
    print("\n" + "=" * 60)
    print("Exercise 1.2: Enhanced Question-Answering")
    print("=" * 60)

    try:
        # Create a ChainOfThought module for better reasoning
        qa_module = dspy.ChainOfThought(DetailedQASignature)

        # Test with context-based question
        test_data = {
            "question": "What programming language is DSPy built with?",
            "context": "DSPy is a framework for algorithmically optimizing LM prompts and weights. It is implemented in Python and provides a clean interface for working with language models.",
        }

        result = qa_module(**test_data)

        print(f"Question: {test_data['question']}")
        print(f"Context: {test_data['context'][:100]}...")
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence}")

        # Validate the result
        validation = validate_signature_output(
            DetailedQASignature,
            test_data,
            validation_rules={
                "answer": lambda x: "python" in x.lower(),  # Should mention Python
                "confidence": lambda x: 0 <= float(x) <= 1,  # Confidence should be 0-1
            },
        )

        print(f"Validation: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}")

        return {
            "exercise": "1.2",
            "success": True,
            "result": result,
            "validation": validation,
        }

    except Exception as e:
        logger.error(f"Exercise 1.2 failed: {e}")
        return {"exercise": "1.2", "success": False, "error": str(e)}


def exercise_1_3_text_classification():
    """
    Exercise 1.3: Text Classification

    This exercise demonstrates:
    - Classification tasks with predefined categories
    - Reasoning and explanation generation
    - Handling categorical outputs
    """
    print("\n" + "=" * 60)
    print("Exercise 1.3: Text Classification")
    print("=" * 60)

    try:
        # Create a ChainOfThought module for classification
        classifier = dspy.ChainOfThought(TextClassificationSignature)

        # Test with a sample text
        test_data = {
            "text": "I love using this new framework! It makes my development work so much easier and more efficient.",
            "categories": "positive, negative, neutral, technical, personal",
        }

        result = classifier(**test_data)

        print(f"Text: {test_data['text']}")
        print(f"Available Categories: {test_data['categories']}")
        print(f"Predicted Category: {result.category}")
        print(f"Reasoning: {result.reasoning}")

        # Validate the result
        available_categories = [
            cat.strip().lower() for cat in test_data["categories"].split(",")
        ]
        validation = validate_signature_output(
            TextClassificationSignature,
            test_data,
            validation_rules={
                "category": lambda x: x.lower().strip() in available_categories,
                "reasoning": lambda x: len(x.strip())
                > 10,  # Reasoning should be substantial
            },
        )

        print(f"Validation: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}")

        return {
            "exercise": "1.3",
            "success": True,
            "result": result,
            "validation": validation,
        }

    except Exception as e:
        logger.error(f"Exercise 1.3 failed: {e}")
        return {"exercise": "1.3", "success": False, "error": str(e)}


def exercise_1_4_sentiment_analysis():
    """
    Exercise 1.4: Sentiment Analysis

    This exercise demonstrates:
    - Combining categorical and numerical outputs
    - Extracting key features from text
    - Multi-faceted analysis tasks
    """
    print("\n" + "=" * 60)
    print("Exercise 1.4: Sentiment Analysis")
    print("=" * 60)

    try:
        # Create a sentiment analysis module
        sentiment_analyzer = dspy.ChainOfThought(SentimentAnalysisSignature)

        # Test with different sentiment examples
        test_cases = [
            "This product is absolutely amazing! I couldn't be happier with my purchase.",
            "The service was terrible and the staff was rude. Very disappointing experience.",
            "The weather is okay today. Nothing special, just average conditions.",
        ]

        results = []
        for text in test_cases:
            result = sentiment_analyzer(text=text)
            results.append(result)

            print(f"\nText: {text}")
            print(f"Sentiment: {result.sentiment}")
            print(f"Score: {result.score}")
            print(f"Keywords: {result.keywords}")

            # Validate the result
            validation = validate_signature_output(
                SentimentAnalysisSignature,
                {"text": text},
                validation_rules={
                    "sentiment": lambda x: x.lower()
                    in ["positive", "negative", "neutral"],
                    "score": lambda x: -1 <= float(x) <= 1,
                    "keywords": lambda x: len(x.strip()) > 0,
                },
            )

            print(f"Validation: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}")

        return {
            "exercise": "1.4",
            "success": True,
            "results": results,
            "test_cases": len(test_cases),
        }

    except Exception as e:
        logger.error(f"Exercise 1.4 failed: {e}")
        return {"exercise": "1.4", "success": False, "error": str(e)}


def exercise_1_5_text_summarization():
    """
    Exercise 1.5: Text Summarization

    This exercise demonstrates:
    - Handling variable-length inputs and outputs
    - Working with formatting constraints
    - Extracting structured information
    """
    print("\n" + "=" * 60)
    print("Exercise 1.5: Text Summarization")
    print("=" * 60)

    try:
        # Create a summarization module
        summarizer = dspy.ChainOfThought(TextSummarizationSignature)

        # Test with a longer text
        test_text = """
        Artificial Intelligence (AI) has revolutionized numerous industries and aspects of daily life. 
        From healthcare to transportation, AI technologies are being integrated to improve efficiency, 
        accuracy, and decision-making processes. Machine learning algorithms can analyze vast amounts 
        of data to identify patterns and make predictions that would be impossible for humans to detect 
        manually. Natural language processing enables computers to understand and generate human language, 
        facilitating better human-computer interactions. Computer vision allows machines to interpret 
        and analyze visual information, leading to advancements in autonomous vehicles, medical imaging, 
        and security systems. However, the rapid advancement of AI also raises important ethical 
        considerations regarding privacy, job displacement, and the need for responsible AI development. 
        As we continue to integrate AI into society, it's crucial to balance innovation with ethical 
        responsibility and ensure that AI benefits all members of society.
        """

        test_data = {"text": test_text.strip(), "max_length": 50}

        result = summarizer(**test_data)

        print(f"Original Text Length: {len(test_text.split())} words")
        print(f"Max Summary Length: {test_data['max_length']} words")
        print(f"\nSummary: {result.summary}")
        print(f"Key Points: {result.key_points}")

        # Validate the result
        summary_word_count = len(result.summary.split())
        validation = validate_signature_output(
            TextSummarizationSignature,
            test_data,
            validation_rules={
                "summary": lambda x: len(x.split())
                <= test_data["max_length"] * 1.2,  # Allow 20% tolerance
                "key_points": lambda x: "•" in x
                or "-" in x
                or "\n" in x,  # Should be formatted as points
            },
        )

        print(f"\nSummary Word Count: {summary_word_count}")
        print(f"Validation: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}")

        return {
            "exercise": "1.5",
            "success": True,
            "result": result,
            "validation": validation,
            "summary_length": summary_word_count,
        }

    except Exception as e:
        logger.error(f"Exercise 1.5 failed: {e}")
        return {"exercise": "1.5", "success": False, "error": str(e)}


def run_all_exercises():
    """Run all Module 1 exercises and return comprehensive results"""
    print("🚀 DSPy Module 1: Foundations - Complete Solutions")
    print("=" * 80)

    # Setup DSPy environment
    if not setup_dspy_environment():
        print("❌ Failed to setup DSPy environment")
        return {"success": False, "error": "Environment setup failed"}

    # Run all exercises
    exercises = [
        exercise_1_1_basic_signature,
        exercise_1_2_enhanced_qa,
        exercise_1_3_text_classification,
        exercise_1_4_sentiment_analysis,
        exercise_1_5_text_summarization,
    ]

    results = []
    successful_exercises = 0

    for exercise_func in exercises:
        try:
            result = exercise_func()
            results.append(result)
            if result.get("success", False):
                successful_exercises += 1
        except Exception as e:
            logger.error(f"Exercise {exercise_func.__name__} failed: {e}")
            results.append(
                {"exercise": exercise_func.__name__, "success": False, "error": str(e)}
            )

    # Summary
    print("\n" + "=" * 80)
    print("MODULE 1 SUMMARY")
    print("=" * 80)
    print(f"Total Exercises: {len(exercises)}")
    print(f"Successful: {successful_exercises}")
    print(f"Failed: {len(exercises) - successful_exercises}")
    print(f"Success Rate: {(successful_exercises / len(exercises)) * 100:.1f}%")

    # Detailed results
    print("\nDetailed Results:")
    for result in results:
        status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
        exercise_name = result.get("exercise", "Unknown")
        print(f"  Exercise {exercise_name}: {status}")
        if not result.get("success", False) and "error" in result:
            print(f"    Error: {result['error']}")

    print("\n🎯 Key Learning Points Demonstrated:")
    print("  • Basic DSPy signature creation and structure")
    print("  • Input and output field definitions with descriptions")
    print("  • Using Predict and ChainOfThought modules")
    print("  • Handling different data types (text, numbers, categories)")
    print("  • Validation and error handling patterns")
    print("  • Multi-input and multi-output signature designs")
    print("  • Context-aware question answering")
    print("  • Classification and sentiment analysis tasks")
    print("  • Text summarization with constraints")

    return {
        "success": successful_exercises == len(exercises),
        "total_exercises": len(exercises),
        "successful_exercises": successful_exercises,
        "results": results,
    }


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="DSPy Module 1 Solutions")
    parser.add_argument(
        "--exercise", type=str, help="Run specific exercise (1.1, 1.2, etc.)"
    )
    parser.add_argument("--all", action="store_true", help="Run all exercises")

    args = parser.parse_args()

    if args.exercise:
        exercise_map = {
            "1.1": exercise_1_1_basic_signature,
            "1.2": exercise_1_2_enhanced_qa,
            "1.3": exercise_1_3_text_classification,
            "1.4": exercise_1_4_sentiment_analysis,
            "1.5": exercise_1_5_text_summarization,
        }

        if args.exercise in exercise_map:
            setup_dspy_environment()
            exercise_map[args.exercise]()
        else:
            print(f"Unknown exercise: {args.exercise}")
            print(f"Available exercises: {', '.join(exercise_map.keys())}")

    elif args.all or len(sys.argv) == 1:
        run_all_exercises()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
