#!/usr/bin/env python3
"""
Reusable DSPy Component Library

This module provides a comprehensive library of reusable DSPy components and domain-specific
module collections. It includes pre-built components for common tasks, composition utilities,
and performance optimization tools.

Learning Objectives:
- Understand component-based architecture in DSPy
- Learn to compose complex systems from reusable components
- Master domain-specific module patterns and best practices
- Implement performance optimization for component systems

Author: DSPy Learning Framework
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import dspy
from dspy import Example

# DSPy configuration will be done after logger is set up

# Import from our custom module framework
try:
    from .custom_module_template import (
        CustomModuleBase,
        ModuleMetadata,
        ModuleValidator,
    )
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent))
    from custom_module_template import CustomModuleBase, ModuleMetadata, ModuleValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure DSPy with available API keys
from dspy_config import configure_dspy_lm, get_configured_model_info

dspy_configured = configure_dspy_lm()
model_info = get_configured_model_info()
if dspy_configured:
    logger.info(f"DSPy configured with {model_info['provider']} {model_info['model']}")


class ComponentType(Enum):
    """Types of reusable components"""

    TEXT_PROCESSOR = "text_processor"
    CLASSIFIER = "classifier"
    GENERATOR = "generator"
    ANALYZER = "analyzer"
    TRANSFORMER = "transformer"
    VALIDATOR = "validator"
    AGGREGATOR = "aggregator"
    ROUTER = "router"


@dataclass
class ComponentConfig:
    """Configuration for reusable components"""

    name: str
    component_type: ComponentType
    parameters: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    performance_requirements: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"


class ReusableComponent(CustomModuleBase, ABC):
    """Base class for all reusable components"""

    def __init__(self, config: ComponentConfig, metadata: ModuleMetadata | None = None):
        if metadata is None:
            metadata = ModuleMetadata(
                name=config.name,
                description=f"Reusable {config.component_type.value} component",
                version=config.version,
                author="DSPy Component Library",
            )
        super().__init__(metadata)
        self.config = config
        self._component_id = (
            f"{config.component_type.value}_{config.name}_{int(time.time())}"
        )
        self._dependencies_resolved = False

    @abstractmethod
    def process(self, **kwargs) -> Any:
        """Main processing method - must be implemented by subclasses"""
        pass

    def forward(self, **kwargs) -> Any:
        """Forward pass delegates to process method"""
        if not self._dependencies_resolved:
            self._resolve_dependencies()
        return self.process(**kwargs)

    def _resolve_dependencies(self):
        """Resolve component dependencies"""
        # In a real implementation, this would resolve actual dependencies
        logger.info(
            f"Resolving dependencies for {self._component_id}: {self.config.dependencies}"
        )
        self._dependencies_resolved = True

    def get_component_info(self) -> dict[str, Any]:
        """Get comprehensive component information"""
        return {
            "component_id": self._component_id,
            "config": {
                "name": self.config.name,
                "type": self.config.component_type.value,
                "parameters": self.config.parameters,
                "dependencies": self.config.dependencies,
                "version": self.config.version,
            },
            "metadata": self.get_info(),
            "dependencies_resolved": self._dependencies_resolved,
        }


# Text Processing Components


class TextCleanerComponent(ReusableComponent):
    """Component for cleaning and preprocessing text"""

    def __init__(self, config: ComponentConfig | None = None):
        if config is None:
            config = ComponentConfig(
                name="text_cleaner",
                component_type=ComponentType.TEXT_PROCESSOR,
                parameters={
                    "remove_html": True,
                    "normalize_whitespace": True,
                    "lowercase": False,
                    "remove_punctuation": False,
                },
            )
        super().__init__(config)
        self._initialized = True

    def process(self, **kwargs) -> dict[str, Any]:
        """Clean and preprocess text"""
        text = kwargs.get("text", "")

        if self.config.parameters.get("remove_html", False):
            import re

            text = re.sub(r"<[^>]+>", "", text)

        if self.config.parameters.get("normalize_whitespace", False):
            import re

            text = re.sub(r"\s+", " ", text).strip()

        if self.config.parameters.get("lowercase", False):
            text = text.lower()

        if self.config.parameters.get("remove_punctuation", False):
            import string

            text = text.translate(str.maketrans("", "", string.punctuation))

        return {
            "cleaned_text": text,
            "original_length": len(kwargs.get("text", "")),
            "cleaned_length": len(text),
            "operations_applied": [k for k, v in self.config.parameters.items() if v],
        }


class TextSummarizerComponent(ReusableComponent):
    """Component for text summarization"""

    def __init__(self, config: ComponentConfig | None = None):
        if config is None:
            config = ComponentConfig(
                name="text_summarizer",
                component_type=ComponentType.GENERATOR,
                parameters={
                    "max_length": 150,
                    "style": "concise",
                    "preserve_key_points": True,
                },
            )
        super().__init__(config)

        # Create DSPy signature for summarization
        class SummarizationSignature(dspy.Signature):
            """Summarize the given text while preserving key information"""

            text = dspy.InputField(desc="Text to summarize")
            max_length = dspy.InputField(desc="Maximum length of summary")
            summary = dspy.OutputField(desc="Concise summary of the text")

        self.summarizer = dspy.ChainOfThought(SummarizationSignature)
        self._initialized = True
        self._fallback_logged = False

    def process(self, **kwargs) -> dict[str, Any]:
        """Generate text summary"""
        text = kwargs.get("text", "")
        max_length = kwargs.get(
            "max_length", self.config.parameters.get("max_length", 150)
        )

        try:
            # Use DSPy for actual summarization
            result = self.summarizer(text=text, max_length=str(max_length))
            summary = result.summary
        except Exception as _:
            # Fallback to simple truncation if DSPy fails
            if not self._fallback_logged:
                logger.info("Using fallback summarization (DSPy LM not configured)")
                self._fallback_logged = True
            words = text.split()
            if len(words) > max_length // 5:  # Rough word count estimation
                summary = " ".join(words[: max_length // 5]) + "..."
            else:
                summary = text

        return {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / len(text) if text else 0,
            "method": "dspy" if "result" in locals() else "fallback",
        }


# Classification Components


class TextClassifierComponent(ReusableComponent):
    """Component for text classification"""

    def __init__(self, categories: list[str], config: ComponentConfig | None = None):
        if config is None:
            config = ComponentConfig(
                name="text_classifier",
                component_type=ComponentType.CLASSIFIER,
                parameters={
                    "categories": categories,
                    "confidence_threshold": 0.7,
                    "return_probabilities": True,
                },
            )
        super().__init__(config)
        self.categories = categories

        # Create DSPy signature for classification
        class ClassificationSignature(dspy.Signature):
            """Classify the given text into one of the specified categories"""

            text = dspy.InputField(desc="Text to classify")
            categories = dspy.InputField(desc="Available categories")
            category = dspy.OutputField(desc="Most appropriate category")
            confidence = dspy.OutputField(desc="Confidence score (0-1)")

        self.classifier = dspy.ChainOfThought(ClassificationSignature)
        self._initialized = True
        self._fallback_logged = False

    def process(self, **kwargs) -> dict[str, Any]:
        """Classify text into categories"""
        text = kwargs.get("text", "")
        categories_str = ", ".join(self.categories)

        try:
            # Use DSPy for classification
            result = self.classifier(text=text, categories=categories_str)
            predicted_category = result.category
            confidence = (
                float(result.confidence)
                if result.confidence.replace(".", "").isdigit()
                else 0.5
            )
        except Exception as _:
            # Fallback to simple keyword-based classification
            if not self._fallback_logged:
                logger.info("Using fallback classification (DSPy LM not configured)")
                self._fallback_logged = True
            predicted_category = self._fallback_classify(text)
            confidence = 0.3  # Low confidence for fallback

        return {
            "predicted_category": predicted_category,
            "confidence": confidence,
            "all_categories": self.categories,
            "text_length": len(text),
            "method": "dspy" if "result" in locals() else "fallback",
        }

    def _fallback_classify(self, text: str) -> str:
        """Simple fallback classification based on keywords"""
        text_lower = text.lower()

        # Simple keyword matching (this would be more sophisticated in practice)
        category_keywords = {
            "technology": [
                "tech",
                "computer",
                "software",
                "digital",
                "ai",
                "machine learning",
            ],
            "business": ["company", "market", "profit", "revenue", "customer", "sales"],
            "science": [
                "research",
                "study",
                "experiment",
                "data",
                "analysis",
                "hypothesis",
            ],
            "sports": ["game", "team", "player", "score", "match", "championship"],
            "entertainment": ["movie", "music", "show", "actor", "celebrity", "film"],
        }

        scores = {}
        for category in self.categories:
            category_lower = category.lower()
            if category_lower in category_keywords:
                keywords = category_keywords[category_lower]
                score = sum(1 for keyword in keywords if keyword in text_lower)
                scores[category] = score
            else:
                # If category not in predefined keywords, check for direct mention
                scores[category] = 1 if category_lower in text_lower else 0

        return max(scores, key=scores.get) if scores else self.categories[0]


# Analysis Components


class SentimentAnalyzerComponent(ReusableComponent):
    """Component for sentiment analysis"""

    def __init__(self, config: ComponentConfig | None = None):
        if config is None:
            config = ComponentConfig(
                name="sentiment_analyzer",
                component_type=ComponentType.ANALYZER,
                parameters={
                    "scale": "positive_negative_neutral",
                    "include_confidence": True,
                    "detailed_emotions": False,
                },
            )
        super().__init__(config)

        # Create DSPy signature for sentiment analysis
        class SentimentSignature(dspy.Signature):
            """Analyze the sentiment of the given text"""

            text = dspy.InputField(desc="Text to analyze for sentiment")
            sentiment = dspy.OutputField(
                desc="Sentiment: positive, negative, or neutral"
            )
            confidence = dspy.OutputField(desc="Confidence score (0-1)")
            reasoning = dspy.OutputField(desc="Brief explanation of the sentiment")

        self.analyzer = dspy.ChainOfThought(SentimentSignature)
        self._initialized = True
        self._fallback_logged = False

    def process(self, **kwargs) -> dict[str, Any]:
        """Analyze sentiment of text"""
        text = kwargs.get("text", "")

        try:
            # Use DSPy for sentiment analysis
            result = self.analyzer(text=text)
            sentiment = result.sentiment.lower()
            confidence = (
                float(result.confidence)
                if result.confidence.replace(".", "").isdigit()
                else 0.5
            )
            reasoning = result.reasoning
        except Exception as _:
            # Fallback to simple sentiment analysis
            if not self._fallback_logged:
                logger.info(
                    "Using fallback sentiment analysis (DSPy LM not configured)"
                )
                self._fallback_logged = True
            sentiment, confidence = self._fallback_sentiment(text)
            reasoning = "Fallback analysis based on keyword matching"

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "reasoning": reasoning,
            "text_length": len(text),
            "method": "dspy" if "result" in locals() else "fallback",
        }

    def _fallback_sentiment(self, text: str) -> tuple:
        """Simple fallback sentiment analysis"""
        positive_words = [
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "love",
            "like",
            "happy",
            "pleased",
        ]
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "horrible",
            "hate",
            "dislike",
            "sad",
            "angry",
            "disappointed",
            "frustrated",
        ]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive", min(0.8, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            return "negative", min(0.8, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            return "neutral", 0.5


# Composition and Integration Components


class ComponentPipeline(ReusableComponent):
    """Component for creating processing pipelines"""

    def __init__(
        self,
        components: list[ReusableComponent],
        config: ComponentConfig | None = None,
    ):
        if config is None:
            config = ComponentConfig(
                name="component_pipeline",
                component_type=ComponentType.AGGREGATOR,
                parameters={
                    "parallel_execution": False,
                    "error_handling": "stop_on_error",
                    "result_aggregation": "sequential",
                },
            )
        super().__init__(config)
        self.components = components
        self._initialized = True

    def process(self, **kwargs) -> dict[str, Any]:
        """Execute pipeline of components"""
        results = []
        current_input = kwargs.copy()

        for i, component in enumerate(self.components):
            try:
                start_time = time.time()
                result = component(**current_input)
                duration = time.time() - start_time

                step_result = {
                    "step": i,
                    "component": component.__class__.__name__,
                    "component_id": getattr(component, "_component_id", f"unknown_{i}"),
                    "result": result,
                    "duration": duration,
                    "success": True,
                }
                results.append(step_result)

                # Update input for next component
                if isinstance(result, dict):
                    current_input.update(result)
                else:
                    current_input["previous_result"] = result

                logger.info(f"Pipeline step {i} completed in {duration:.3f}s")

            except Exception as e:
                error_result = {
                    "step": i,
                    "component": component.__class__.__name__,
                    "error": str(e),
                    "success": False,
                }
                results.append(error_result)

                if self.config.parameters.get("error_handling") == "stop_on_error":
                    logger.error("Pipeline stopped at step %d due to error: %s", i, e)
                    break
                else:
                    logger.warning("Pipeline step %d failed, continuing: %s", i, e)

        return {
            "pipeline_results": results,
            "total_steps": len(self.components),
            "successful_steps": sum(1 for r in results if r.get("success", False)),
            "failed_steps": sum(1 for r in results if not r.get("success", True)),
            "total_duration": sum(r.get("duration", 0) for r in results),
            "final_result": (
                results[-1]["result"]
                if results and results[-1].get("success")
                else None
            ),
        }


class ComponentRouter(ReusableComponent):
    """Component for routing inputs to different components based on conditions"""

    def __init__(
        self,
        routes: dict[str, tuple],
        default_component: ReusableComponent | None = None,
        config: ComponentConfig | None = None,
    ):
        """
        Initialize router with routes.

        Args:
            routes: dict mapping route names to (condition_func, component) tuples
            default_component: Component to use if no conditions match
        """
        if config is None:
            config = ComponentConfig(
                name="component_router",
                component_type=ComponentType.ROUTER,
                parameters={
                    "route_count": len(routes),
                    "has_default": default_component is not None,
                },
            )
        super().__init__(config)
        self.routes = routes
        self.default_component = default_component
        self._initialized = True

    def process(self, **kwargs) -> dict[str, Any]:
        """Route input to appropriate component"""
        selected_route = None
        selected_component = None

        # Check each route condition
        for route_name, (condition_func, component) in self.routes.items():
            try:
                if condition_func(**kwargs):
                    selected_route = route_name
                    selected_component = component
                    break
            except Exception as e:
                logger.warning(f"Error evaluating route condition '{route_name}': {e}")

        # Use default if no route matched
        if selected_component is None:
            if self.default_component:
                selected_route = "default"
                selected_component = self.default_component
            else:
                return {
                    "error": "No matching route found and no default component specified",
                    "available_routes": list(self.routes.keys()),
                    "input_received": kwargs,
                }

        # Execute selected component
        try:
            start_time = time.time()
            result = selected_component(**kwargs)
            duration = time.time() - start_time

            return {
                "selected_route": selected_route,
                "component_used": selected_component.__class__.__name__,
                "result": result,
                "execution_time": duration,
                "success": True,
            }
        except Exception as e:
            return {
                "selected_route": selected_route,
                "component_used": selected_component.__class__.__name__,
                "error": str(e),
                "success": False,
            }


# Domain-Specific Component Collections


class NLPComponentSuite:
    """Suite of NLP-focused components"""

    def __init__(self):
        self.text_cleaner = TextCleanerComponent()
        self.summarizer = TextSummarizerComponent()
        self.classifier = TextClassifierComponent(
            ["news", "opinion", "technical", "creative"]
        )
        self.sentiment_analyzer = SentimentAnalyzerComponent()

    def create_analysis_pipeline(self) -> ComponentPipeline:
        """Create a comprehensive text analysis pipeline"""
        components = [
            self.text_cleaner,
            self.sentiment_analyzer,
            self.classifier,
            self.summarizer,
        ]

        config = ComponentConfig(
            name="nlp_analysis_pipeline",
            component_type=ComponentType.AGGREGATOR,
            parameters={"domain": "nlp", "comprehensive": True},
        )

        return ComponentPipeline(components, config)

    def create_content_router(self) -> ComponentRouter:
        """Create a router for different types of content processing"""
        routes = {
            "long_text": (
                lambda **kwargs: len(kwargs.get("text", "")) > 1000,
                self.summarizer,
            ),
            "short_text": (
                lambda **kwargs: len(kwargs.get("text", "")) <= 1000,
                self.sentiment_analyzer,
            ),
        }

        config = ComponentConfig(
            name="content_router",
            component_type=ComponentType.ROUTER,
            parameters={"domain": "nlp", "routing_strategy": "length_based"},
        )

        return ComponentRouter(routes, self.text_cleaner, config)


class BusinessAnalyticsComponentSuite:
    """Suite of business analytics components"""

    def __init__(self):
        # Create business-specific classifiers
        self.document_classifier = TextClassifierComponent(
            ["contract", "report", "email", "proposal", "invoice"]
        )
        self.priority_classifier = TextClassifierComponent(
            ["urgent", "high", "medium", "low"]
        )
        self.sentiment_analyzer = SentimentAnalyzerComponent()

    def create_document_processing_pipeline(self) -> ComponentPipeline:
        """Create a business document processing pipeline"""
        components = [
            TextCleanerComponent(),  # Clean the document
            self.document_classifier,  # Classify document type
            self.priority_classifier,  # Determine priority
            self.sentiment_analyzer,  # Analyze sentiment
        ]

        config = ComponentConfig(
            name="business_document_pipeline",
            component_type=ComponentType.AGGREGATOR,
            parameters={"domain": "business", "document_focused": True},
        )

        return ComponentPipeline(components, config)


# Performance Optimization Tools


class ComponentPerformanceOptimizer:
    """Tools for optimizing component performance"""

    @staticmethod
    def benchmark_component(
        component: ReusableComponent,
        test_inputs: list[dict[str, Any]],
        iterations: int = 10,
    ) -> dict[str, Any]:
        """Benchmark component performance"""
        results = {
            "component_name": component.__class__.__name__,
            "test_count": len(test_inputs),
            "iterations": iterations,
            "execution_times": [],
            "success_count": 0,
            "error_count": 0,
            "errors": [],
        }

        for iteration in range(iterations):
            for i, test_input in enumerate(test_inputs):
                start_time = time.time()
                try:
                    result = component(**test_input)
                    execution_time = time.time() - start_time
                    results["execution_times"].append(execution_time)
                    results["success_count"] += 1
                except Exception as e:
                    execution_time = time.time() - start_time
                    results["execution_times"].append(execution_time)
                    results["error_count"] += 1
                    results["errors"].append(
                        {"iteration": iteration, "test_index": i, "error": str(e)}
                    )

        # Calculate statistics
        if results["execution_times"]:
            results["avg_execution_time"] = sum(results["execution_times"]) / len(
                results["execution_times"]
            )
            results["min_execution_time"] = min(results["execution_times"])
            results["max_execution_time"] = max(results["execution_times"])
            results["success_rate"] = results["success_count"] / (
                results["success_count"] + results["error_count"]
            )

        return results

    @staticmethod
    def optimize_pipeline(
        pipeline: ComponentPipeline, test_inputs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze and suggest optimizations for a component pipeline"""
        # Benchmark the pipeline
        benchmark_results = ComponentPerformanceOptimizer.benchmark_component(
            pipeline, test_inputs
        )

        # Analyze individual components
        component_benchmarks = []
        for component in pipeline.components:
            component_benchmark = ComponentPerformanceOptimizer.benchmark_component(
                component, test_inputs, 3
            )
            component_benchmarks.append(component_benchmark)

        # Generate optimization suggestions
        suggestions = []

        # Check for slow components
        if component_benchmarks:
            avg_times = [cb.get("avg_execution_time", 0) for cb in component_benchmarks]
            if avg_times:
                max_time = max(avg_times)
                avg_time = sum(avg_times) / len(avg_times)

                for i, cb in enumerate(component_benchmarks):
                    if cb.get("avg_execution_time", 0) > avg_time * 2:
                        suggestions.append(
                            f"Component {i} ({cb['component_name']}) is significantly slower than average"
                        )

        # Check error rates
        for i, cb in enumerate(component_benchmarks):
            if cb.get("error_count", 0) > 0:
                error_rate = cb["error_count"] / (
                    cb["success_count"] + cb["error_count"]
                )
                if error_rate > 0.1:  # More than 10% error rate
                    suggestions.append(
                        f"Component {i} ({cb['component_name']}) has high error rate: {error_rate:.2%}"
                    )

        return {
            "pipeline_benchmark": benchmark_results,
            "component_benchmarks": component_benchmarks,
            "optimization_suggestions": suggestions,
            "total_components": len(pipeline.components),
        }


# Component Registry and Management


class ComponentRegistry:
    """Registry for managing reusable components"""

    def __init__(self):
        self._components = {}
        self._component_types = {}
        self._performance_cache = {}

    def register_component(
        self,
        name: str,
        component_class: type[ReusableComponent],
        component_type: ComponentType,
    ):
        """Register a component class"""
        self._components[name] = component_class
        self._component_types[name] = component_type
        logger.info(f"Registered component: {name} ({component_type.value})")

    def create_component(self, name: str, **kwargs) -> ReusableComponent:
        """Create a component instance"""
        if name not in self._components:
            raise ValueError(f"Unknown component: {name}")

        component_class = self._components[name]
        return component_class(**kwargs)

    def list_components(self, component_type: ComponentType | None = None) -> list[str]:
        """list available components, optionally filtered by type"""
        if component_type is None:
            return list(self._components.keys())
        else:
            return [
                name
                for name, ctype in self._component_types.items()
                if ctype == component_type
            ]

    def get_component_info(self, name: str) -> dict[str, Any]:
        """Get information about a registered component"""
        if name not in self._components:
            raise ValueError(f"Unknown component: {name}")

        component_class = self._components[name]
        return {
            "name": name,
            "class": component_class.__name__,
            "type": self._component_types[name].value,
            "module": component_class.__module__,
            "docstring": component_class.__doc__,
        }

    def benchmark_component(
        self, name: str, test_inputs: list[dict[str, Any]], **component_kwargs
    ) -> dict[str, Any]:
        """Benchmark a component and cache results"""
        cache_key = f"{name}_{hash(str(sorted(component_kwargs.items())))}"

        if cache_key in self._performance_cache:
            logger.info(f"Using cached benchmark results for {name}")
            return self._performance_cache[cache_key]

        component = self.create_component(name, **component_kwargs)
        results = ComponentPerformanceOptimizer.benchmark_component(
            component, test_inputs
        )

        self._performance_cache[cache_key] = results
        return results


# Initialize default registry
default_registry = ComponentRegistry()

# Register built-in components
default_registry.register_component(
    "text_cleaner", TextCleanerComponent, ComponentType.TEXT_PROCESSOR
)
default_registry.register_component(
    "text_summarizer", TextSummarizerComponent, ComponentType.GENERATOR
)
default_registry.register_component(
    "text_classifier", TextClassifierComponent, ComponentType.CLASSIFIER
)
default_registry.register_component(
    "sentiment_analyzer", SentimentAnalyzerComponent, ComponentType.ANALYZER
)
default_registry.register_component(
    "component_pipeline", ComponentPipeline, ComponentType.AGGREGATOR
)
default_registry.register_component(
    "component_router", ComponentRouter, ComponentType.ROUTER
)


def demonstrate_component_library():
    """Demonstrate the reusable component library"""
    print("=== Reusable DSPy Component Library Demonstration ===\n")

    # Example 1: Individual components
    print("1. Individual Component Usage:")
    print("-" * 40)

    # Create and test text cleaner
    cleaner = TextCleanerComponent()
    sample_text = "<p>Hello, World!   This is a   test.</p>"

    cleaned_result = cleaner(text=sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned_result['cleaned_text']}")
    print(f"Operations: {cleaned_result['operations_applied']}")

    # Create and test sentiment analyzer
    print("\n2. Sentiment Analysis:")
    print("-" * 40)

    sentiment_analyzer = SentimentAnalyzerComponent()
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible and I hate it.",
        "It's okay, nothing special.",
    ]

    for text in test_texts:
        result = sentiment_analyzer(text=text)
        print(f"Text: {text}")
        print(
            f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})"
        )
        print(f"Reasoning: {result['reasoning']}")
        print()

    # Example 3: Component Pipeline
    print("3. Component Pipeline:")
    print("-" * 40)

    # Create NLP suite and pipeline
    nlp_suite = NLPComponentSuite()
    pipeline = nlp_suite.create_analysis_pipeline()

    test_text = "This is an amazing technical article about artificial intelligence and machine learning. The content is very informative and well-written, providing excellent insights into the latest developments in AI technology."

    pipeline_result = pipeline(text=test_text)
    print(f"Input: {test_text[:100]}...")
    print(f"Pipeline steps: {pipeline_result['total_steps']}")
    print(f"Successful steps: {pipeline_result['successful_steps']}")
    print(f"Total duration: {pipeline_result['total_duration']:.3f}s")

    if pipeline_result["final_result"]:
        print(f"Final result keys: {list(pipeline_result['final_result'].keys())}")

    # Example 4: Component Router
    print("\n4. Component Router:")
    print("-" * 40)

    router = nlp_suite.create_content_router()

    short_text = "Great product!"
    long_text = (
        "This is a very long text that contains multiple sentences and paragraphs. "
        * 20
    )

    for text, label in [(short_text, "short"), (long_text, "long")]:
        result = router(text=text)
        print(f"{label.title()} text ({len(text)} chars):")
        print(f"Selected route: {result['selected_route']}")
        print(f"Component used: {result['component_used']}")
        print(f"Execution time: {result['execution_time']:.3f}s")
        print()

    # Example 5: Component Registry
    print("5. Component Registry:")
    print("-" * 40)

    print("Available components:")
    for component_name in default_registry.list_components():
        info = default_registry.get_component_info(component_name)
        print(f"  • {component_name}: {info['type']}")

    # Example 6: Performance Optimization
    print("\n6. Performance Optimization:")
    print("-" * 40)

    # Create a simple pipeline for benchmarking
    simple_pipeline = ComponentPipeline(
        [TextCleanerComponent(), SentimentAnalyzerComponent()]
    )

    test_inputs = [
        {"text": "This is a test sentence."},
        {"text": "Another test with different content."},
        {"text": "Final test message for benchmarking."},
    ]

    optimization_results = ComponentPerformanceOptimizer.optimize_pipeline(
        simple_pipeline, test_inputs
    )

    print(f"Pipeline benchmark:")
    print(
        f"  Average execution time: {optimization_results['pipeline_benchmark'].get('avg_execution_time', 0):.3f}s"
    )
    print(
        f"  Success rate: {optimization_results['pipeline_benchmark'].get('success_rate', 0):.2%}"
    )

    if optimization_results["optimization_suggestions"]:
        print(f"  Optimization suggestions:")
        for suggestion in optimization_results["optimization_suggestions"]:
            print(f"    - {suggestion}")
    else:
        print(f"  No optimization suggestions - pipeline performing well!")


if __name__ == "__main__":
    """
    Reusable DSPy Component Library Demonstration

    This script demonstrates:
    1. Individual reusable components for common tasks
    2. Component composition and pipeline creation
    3. Domain-specific component suites
    4. Component routing and conditional processing
    5. Performance optimization and benchmarking
    6. Component registry and management
    """

    try:
        demonstrate_component_library()

        print("\n✅ Component library demonstration completed successfully!")
        print("\nKey takeaways:")
        print("- Reusable components provide modular, testable building blocks")
        print("- Component pipelines enable complex processing workflows")
        print("- Domain-specific suites offer pre-configured component collections")
        print("- Component routers enable conditional processing logic")
        print("- Performance optimization tools help identify bottlenecks")
        print("- Component registry provides centralized management")

    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        logger.exception("Component library demonstration failed")
