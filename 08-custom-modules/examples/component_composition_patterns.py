#!/usr/bin/env python3
"""
Component Composition and Pipeline Creation Example

This example demonstrates how to compose multiple components into complex processing
pipelines, including sequential, parallel, and conditional compositions.

Learning Objectives:
- Create complex processing pipelines from simple components
- Implement different composition patterns (sequential, parallel, conditional)
- Handle data flow between components
- Add error handling and recovery mechanisms
- Monitor pipeline performance and optimization

Author: DSPy Learning Framework
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import logging
import time

# Configure DSPy with available API keys
from dspy_config import configure_dspy_lm, get_configured_model_info

dspy_configured = configure_dspy_lm()
if dspy_configured:
    model_info = get_configured_model_info()
    print(f"✅ DSPy configured with {model_info['provider']} {model_info['model']}")
else:
    print("ℹ️  Using fallback implementations (no API keys configured)")

import dspy


# Configure DSPy with a dummy LM for examples
# This allows the examples to work without requiring API keys
class DummyLM(dspy.LM):
    """Dummy Language Model for examples - returns simple mock responses"""

    def __init__(self):
        super().__init__("dummy-model")

    def basic_request(self, prompt, **kwargs):
        # Simple mock responses based on the prompt content
        if "sentiment" in prompt.lower():
            return ["positive\n0.8\nThe text contains positive language and tone."]
        elif "summarize" in prompt.lower():
            return ["This is a brief summary of the provided text content."]
        elif "classify" in prompt.lower() or "category" in prompt.lower():
            return ["technology\n0.7"]
        else:
            return ["Mock response for the given prompt."]


# Configure DSPy to use our dummy LM
dspy.configure(lm=DummyLM())

from component_library import (
    ComponentConfig,
    ComponentPipeline,
    ComponentRouter,
    ComponentType,
    NLPComponentSuite,
    ReusableComponent,
    SentimentAnalyzerComponent,
    TextCleanerComponent,
    TextSummarizerComponent,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeywordExtractorComponent(ReusableComponent):
    """Custom component for extracting keywords from text"""

    def __init__(self, max_keywords: int = 10):
        config = ComponentConfig(
            name="keyword_extractor",
            component_type=ComponentType.ANALYZER,
            parameters={"max_keywords": max_keywords},
        )
        super().__init__(config)
        self.max_keywords = max_keywords
        self._initialized = True

    def process(self, **kwargs) -> dict:
        """Extract keywords from text"""
        text = kwargs.get("text", "")

        if not text:
            return {"keywords": [], "keyword_count": 0}

        # Simple keyword extraction (in practice, you'd use more sophisticated methods)
        import string

        # Remove punctuation and convert to lowercase
        cleaned_text = text.translate(str.maketrans("", "", string.punctuation)).lower()
        words = cleaned_text.split()

        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
        }

        # Filter words and count frequency
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and get top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[
            : self.max_keywords
        ]
        keyword_list = [word for word, freq in keywords]

        return {
            "keywords": keyword_list,
            "keyword_count": len(keyword_list),
            "keyword_frequencies": dict(keywords),
            "total_words_processed": len(words),
            "unique_words": len(word_freq),
        }


class TextStatisticsComponent(ReusableComponent):
    """Component for calculating detailed text statistics"""

    def __init__(self):
        config = ComponentConfig(
            name="text_statistics",
            component_type=ComponentType.ANALYZER,
            parameters={"include_advanced_stats": True},
        )
        super().__init__(config)
        self._initialized = True

    def process(self, **kwargs) -> dict:
        """Calculate comprehensive text statistics"""
        text = kwargs.get("text", "")

        if not text:
            return self._empty_stats()

        import re
        import string

        # Basic counts
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.findall(r"[.!?]+", text))
        paragraph_count = len([p for p in text.split("\n\n") if p.strip()])

        # Character analysis
        letter_count = sum(1 for c in text if c.isalpha())
        digit_count = sum(1 for c in text if c.isdigit())
        punctuation_count = sum(1 for c in text if c in string.punctuation)
        whitespace_count = sum(1 for c in text if c.isspace())

        # Word analysis
        words = text.split()
        if words:
            avg_word_length = sum(
                len(word.strip(string.punctuation)) for word in words
            ) / len(words)
            longest_word = max(words, key=lambda w: len(w.strip(string.punctuation)))
            shortest_word = min(words, key=lambda w: len(w.strip(string.punctuation)))
        else:
            avg_word_length = 0
            longest_word = ""
            shortest_word = ""

        # Sentence analysis
        if sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
        else:
            avg_sentence_length = 0

        # Readability metrics (simplified)
        if sentence_count > 0 and word_count > 0:
            # Simplified Flesch Reading Ease approximation
            avg_sentence_len = word_count / sentence_count
            avg_syllables = avg_word_length * 1.5  # Rough approximation
            flesch_score = 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables)
            flesch_score = max(0, min(100, flesch_score))  # Clamp to 0-100
        else:
            flesch_score = 0

        return {
            "character_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "letter_count": letter_count,
            "digit_count": digit_count,
            "punctuation_count": punctuation_count,
            "whitespace_count": whitespace_count,
            "average_word_length": round(avg_word_length, 2),
            "average_sentence_length": round(avg_sentence_length, 2),
            "longest_word": longest_word.strip(string.punctuation),
            "shortest_word": shortest_word.strip(string.punctuation),
            "flesch_reading_ease": round(flesch_score, 2),
            "readability_level": self._get_readability_level(flesch_score),
        }

    def _empty_stats(self) -> dict:
        """Return empty statistics for empty text"""
        return {
            "character_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "paragraph_count": 0,
            "letter_count": 0,
            "digit_count": 0,
            "punctuation_count": 0,
            "whitespace_count": 0,
            "average_word_length": 0,
            "average_sentence_length": 0,
            "longest_word": "",
            "shortest_word": "",
            "flesch_reading_ease": 0,
            "readability_level": "unknown",
        }

    def _get_readability_level(self, flesch_score: float) -> str:
        """Convert Flesch score to readability level"""
        if flesch_score >= 90:
            return "very_easy"
        elif flesch_score >= 80:
            return "easy"
        elif flesch_score >= 70:
            return "fairly_easy"
        elif flesch_score >= 60:
            return "standard"
        elif flesch_score >= 50:
            return "fairly_difficult"
        elif flesch_score >= 30:
            return "difficult"
        else:
            return "very_difficult"


class AdvancedTextProcessingPipeline(ComponentPipeline):
    """Advanced text processing pipeline with custom components"""

    def __init__(self, include_keywords: bool = True, include_statistics: bool = True):
        # Create components
        components = [TextCleanerComponent(), SentimentAnalyzerComponent()]

        if include_keywords:
            components.append(KeywordExtractorComponent())

        if include_statistics:
            components.append(TextStatisticsComponent())

        components.append(TextSummarizerComponent())

        # Create configuration
        config = ComponentConfig(
            name="advanced_text_processing_pipeline",
            component_type=ComponentType.AGGREGATOR,
            parameters={
                "include_keywords": include_keywords,
                "include_statistics": include_statistics,
                "comprehensive_analysis": True,
            },
        )

        super().__init__(components, config)

    def process(self, **kwargs) -> dict:
        """Process text through the advanced pipeline"""
        # Execute the standard pipeline
        pipeline_result = super().process(**kwargs)

        # Aggregate results from all components
        aggregated_results = {
            "pipeline_info": {
                "total_steps": pipeline_result["total_steps"],
                "successful_steps": pipeline_result["successful_steps"],
                "total_duration": pipeline_result["total_duration"],
            }
        }

        # Extract results from each component
        for step_result in pipeline_result["pipeline_results"]:
            if step_result.get("success") and "result" in step_result:
                component_name = step_result["component"]
                result_data = step_result["result"]

                if isinstance(result_data, dict):
                    # Prefix keys with component name to avoid conflicts
                    prefixed_results = {
                        f"{component_name.lower()}_{key}": value
                        for key, value in result_data.items()
                    }
                    aggregated_results.update(prefixed_results)

        return aggregated_results


class ConditionalTextRouter(ComponentRouter):
    """Router that directs text to different processing paths based on characteristics"""

    def __init__(self):
        # Define routing conditions
        def is_long_text(**kwargs):
            return len(kwargs.get("text", "")) > 500

        def is_technical_text(**kwargs):
            text = kwargs.get("text", "").lower()
            technical_keywords = [
                "algorithm",
                "software",
                "technology",
                "computer",
                "data",
                "system",
            ]
            return any(keyword in text for keyword in technical_keywords)

        def is_emotional_text(**kwargs):
            text = kwargs.get("text", "").lower()
            emotional_keywords = [
                "love",
                "hate",
                "amazing",
                "terrible",
                "wonderful",
                "awful",
            ]
            return any(keyword in text for keyword in emotional_keywords)

        # Create processing components for different paths
        long_text_processor = AdvancedTextProcessingPipeline(
            include_keywords=True, include_statistics=True
        )
        technical_processor = ComponentPipeline(
            [
                TextCleanerComponent(),
                KeywordExtractorComponent(max_keywords=15),
                TextStatisticsComponent(),
            ]
        )
        emotional_processor = ComponentPipeline(
            [
                TextCleanerComponent(),
                SentimentAnalyzerComponent(),
                TextSummarizerComponent(),
            ]
        )
        default_processor = ComponentPipeline(
            [TextCleanerComponent(), SentimentAnalyzerComponent()]
        )

        # Define routes
        routes = {
            "long_text": (is_long_text, long_text_processor),
            "technical_text": (is_technical_text, technical_processor),
            "emotional_text": (is_emotional_text, emotional_processor),
        }

        config = ComponentConfig(
            name="conditional_text_router",
            component_type=ComponentType.ROUTER,
            parameters={
                "routing_strategy": "content_based",
                "routes": list(routes.keys()),
            },
        )

        super().__init__(routes, default_processor, config)


def demonstrate_component_composition():
    """Demonstrate various component composition patterns"""
    print("=== Component Composition and Pipeline Creation Example ===\n")

    # Sample texts for testing
    test_texts = [
        {"name": "Short Simple Text", "text": "Hello, world! This is a simple test."},
        {
            "name": "Technical Text",
            "text": "Machine learning algorithms process data using neural networks and statistical models. The software implementation requires careful consideration of computational complexity and system architecture.",
        },
        {
            "name": "Emotional Text",
            "text": "I absolutely love this amazing product! It's wonderful and has exceeded all my expectations. The experience has been fantastic and I hate to think what I would do without it.",
        },
        {
            "name": "Long Comprehensive Text",
            "text": """Artificial intelligence represents one of the most significant technological advances of our time.
            The field encompasses machine learning, natural language processing, computer vision, and robotics.
            Researchers and practitioners continue to push the boundaries of what's possible with AI systems.

            Modern AI applications span across healthcare, finance, education, transportation, and entertainment.
            Deep learning models have achieved remarkable success in image recognition, language translation,
            and game playing. However, challenges remain in areas such as explainability, bias mitigation,
            and ethical AI development.

            The future of AI holds tremendous promise for solving complex global challenges while raising
            important questions about the role of human intelligence in an increasingly automated world.""",
        },
    ]

    # Example 1: Basic Sequential Pipeline
    print("1. Basic Sequential Pipeline:")
    print("-" * 50)

    basic_pipeline = ComponentPipeline(
        [
            TextCleanerComponent(),
            SentimentAnalyzerComponent(),
            KeywordExtractorComponent(),
        ]
    )

    test_text = test_texts[1]["text"]  # Technical text
    result = basic_pipeline(text=test_text)

    print(f"Input: {test_text[:100]}...")
    print(f"Pipeline Steps: {result['total_steps']}")
    print(f"Successful Steps: {result['successful_steps']}")
    print(f"Total Duration: {result['total_duration']:.3f}s")

    if result["final_result"]:
        final = result["final_result"]
        print(f"Keywords Found: {final.get('keyword_count', 'N/A')}")
        if "keywords" in final:
            print(f"Top Keywords: {', '.join(final['keywords'][:5])}")

    # Example 2: Advanced Processing Pipeline
    print("\n2. Advanced Processing Pipeline:")
    print("-" * 50)

    advanced_pipeline = AdvancedTextProcessingPipeline(
        include_keywords=True, include_statistics=True
    )

    result = advanced_pipeline(text=test_text)

    print(f"Advanced Pipeline Results:")
    print(f"  Steps: {result['pipeline_info']['total_steps']}")
    print(f"  Duration: {result['pipeline_info']['total_duration']:.3f}s")

    # Display aggregated results
    for key, value in result.items():
        if key != "pipeline_info" and not key.startswith("textcleanercomponent"):
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")
            elif isinstance(value, str) and len(value) < 50:
                print(f"  {key}: {value}")
            elif isinstance(value, list) and len(value) < 10:
                print(f"  {key}: {value}")

    # Example 3: Conditional Routing
    print("\n3. Conditional Text Routing:")
    print("-" * 50)

    router = ConditionalTextRouter()

    for test_case in test_texts:
        print(f"\nTesting: {test_case['name']}")
        print(f"Text: {test_case['text'][:80]}...")

        result = router(text=test_case["text"])

        print(f"Selected Route: {result['selected_route']}")
        print(f"Component Used: {result['component_used']}")
        print(f"Execution Time: {result['execution_time']:.3f}s")
        print(f"Success: {result['success']}")

    # Example 4: Parallel Component Execution
    print("\n4. Parallel Component Execution:")
    print("-" * 50)

    from module_composition import ParallelComposition

    # Create parallel composition for independent analyses
    parallel_analyzers = ParallelComposition(
        [
            SentimentAnalyzerComponent(),
            KeywordExtractorComponent(),
            TextStatisticsComponent(),
        ],
        "Parallel Text Analysis",
    )

    test_text = test_texts[3]["text"]  # Long text
    result = parallel_analyzers(text=test_text)

    print(f"Parallel Execution Results:")
    print(f"  Total Steps: {result['total_steps']}")
    print(f"  Successful Steps: {result['successful_steps']}")
    print(f"  Parallel Execution Time: {result['parallel_execution_time']:.3f}s")

    # Show results from each parallel component
    for step_result in result["steps"]:
        if step_result.get("success"):
            component = step_result["module"]
            exec_time = step_result["execution_time"]
            print(f"  {component}: {exec_time:.3f}s")


def demonstrate_error_handling_and_recovery():
    """Demonstrate error handling in component compositions"""
    print("\n5. Error Handling and Recovery:")
    print("-" * 50)

    # Create a component that might fail
    class UnreliableComponent(ReusableComponent):
        def __init__(self, failure_rate: float = 0.3):
            config = ComponentConfig(
                name="unreliable_component", component_type=ComponentType.ANALYZER
            )
            super().__init__(config)
            self.failure_rate = failure_rate
            self._initialized = True

        def process(self, **kwargs):
            import random

            if random.random() < self.failure_rate:
                raise Exception("Simulated component failure")
            return {"status": "success", "processed": True}

    # Test with different error handling strategies
    error_handling_configs = [
        {"error_handling": "stop_on_error"},
        {"error_handling": "continue_on_error"},
    ]

    for config in error_handling_configs:
        print(f"\nTesting with {config['error_handling']}:")

        # Create pipeline with unreliable component
        components = [
            TextCleanerComponent(),
            UnreliableComponent(failure_rate=0.5),  # 50% failure rate
            SentimentAnalyzerComponent(),
        ]

        pipeline_config = ComponentConfig(
            name="error_test_pipeline",
            component_type=ComponentType.AGGREGATOR,
            parameters=config,
        )

        pipeline = ComponentPipeline(components, pipeline_config)

        # Run multiple tests
        successes = 0
        failures = 0

        for i in range(5):
            try:
                result = pipeline(text="Test text for error handling")
                if result["successful_steps"] > 0:
                    successes += 1
                else:
                    failures += 1
                print(
                    f"  Test {i+1}: {result['successful_steps']}/{result['total_steps']} steps successful"
                )
            except Exception as e:
                failures += 1
                print(f"  Test {i+1}: Pipeline failed - {e}")

        print(f"  Overall: {successes} successes, {failures} failures")


def demonstrate_performance_optimization():
    """Demonstrate performance monitoring and optimization"""
    print("\n6. Performance Monitoring and Optimization:")
    print("-" * 50)

    from component_library import ComponentPerformanceOptimizer

    # Create different pipeline configurations
    pipelines = {
        "Basic": ComponentPipeline(
            [TextCleanerComponent(), SentimentAnalyzerComponent()]
        ),
        "Advanced": AdvancedTextProcessingPipeline(),
        "Comprehensive": ComponentPipeline(
            [
                TextCleanerComponent(),
                SentimentAnalyzerComponent(),
                KeywordExtractorComponent(),
                TextStatisticsComponent(),
                TextSummarizerComponent(),
            ]
        ),
    }

    # Test inputs for benchmarking
    test_inputs = [
        {"text": "Short test text."},
        {
            "text": "Medium length test text with multiple sentences and various content types."
        },
        {"text": "Long test text with extensive content. " * 20},
    ]

    print("Performance Comparison:")

    for name, pipeline in pipelines.items():
        print(f"\n{name} Pipeline:")

        # Benchmark the pipeline
        benchmark_results = ComponentPerformanceOptimizer.benchmark_component(
            pipeline, test_inputs, iterations=3
        )

        print(
            f"  Average Execution Time: {benchmark_results.get('avg_execution_time', 0):.4f}s"
        )
        print(f"  Success Rate: {benchmark_results.get('success_rate', 0):.2%}")
        print(
            f"  Total Tests: {benchmark_results['test_count'] * benchmark_results['iterations']}"
        )

        if benchmark_results.get("errors"):
            print(f"  Errors: {len(benchmark_results['errors'])}")

    # Optimization analysis
    print(f"\nOptimization Analysis:")
    comprehensive_pipeline = pipelines["Comprehensive"]
    optimization_results = ComponentPerformanceOptimizer.optimize_pipeline(
        comprehensive_pipeline, test_inputs
    )

    if optimization_results["optimization_suggestions"]:
        print("Suggestions:")
        for suggestion in optimization_results["optimization_suggestions"]:
            print(f"  - {suggestion}")
    else:
        print("  No optimization suggestions - pipeline performing well!")


if __name__ == "__main__":
    """
    Component Composition and Pipeline Creation Example

    This script demonstrates:
    1. Creating custom components for specific tasks
    2. Building sequential processing pipelines
    3. Implementing parallel component execution
    4. Creating conditional routing based on content
    5. Advanced pipeline composition with aggregated results
    6. Error handling and recovery strategies
    7. Performance monitoring and optimization
    """

    try:
        demonstrate_component_composition()
        demonstrate_error_handling_and_recovery()
        demonstrate_performance_optimization()

        print(
            "\n✅ Component Composition and Pipeline Creation example completed successfully!"
        )
        print("\nKey Learning Points:")
        print("- Component composition enables building complex processing systems")
        print("- Different composition patterns serve different use cases")
        print("- Conditional routing allows adaptive processing based on content")
        print("- Parallel execution can improve performance for independent operations")
        print("- Error handling strategies affect system reliability and robustness")
        print("- Performance monitoring helps identify optimization opportunities")
        print("- Aggregated results provide comprehensive analysis capabilities")

    except Exception as e:
        print(f"\n❌ Component Composition and Pipeline Creation example failed: {e}")
        logger.exception(
            "Component Composition and Pipeline Creation example execution failed"
        )
